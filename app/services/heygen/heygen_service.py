"""HeyGen v2 avatar video generation client + orchestrator.

Thin wrapper over HeyGen's REST API (no SDK). Uses stdlib urllib to avoid
adding dependencies. Exposes:

    * generate_raw(payload)            -> submit a full v2 payload
    * generate_simple(request)         -> build the payload from our convenience schema
    * get_status(video_id)             -> poll status
    * wait_for_completion(video_id)    -> block until completed/failed/timeout
    * list_avatars() / list_voices()   -> discovery helpers

Auth: HEYGEN_API_KEY via `X-Api-Key` header.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.core.config import settings
from app.core.logger import get_logger
from app.models.heygen import (
    HeyGenGenerateRawRequest,
    HeyGenGenerateResponse,
    HeyGenGenerateSimpleRequest,
)
from app.utils.file_manager import job_dir, new_job_id
from app.utils.storage import storage_client
from app.utils.video_download import download_to_file

logger = get_logger("services.heygen")


class HeyGenError(RuntimeError):
    """HeyGen API call failed."""


class HeyGenNotConfiguredError(HeyGenError):
    """HEYGEN_API_KEY missing."""


# ----------------------------- low-level HTTP ----------------------------- #


def _require_configured() -> None:
    if not settings.heygen.api_key:
        raise HeyGenNotConfiguredError(
            "HEYGEN_API_KEY is not configured. Set it in your .env."
        )


def _headers(json_body: bool = True) -> dict[str, str]:
    h = {
        "X-Api-Key": settings.heygen.api_key,
        "Accept": "application/json",
    }
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _request(
    method: str,
    path: str,
    *,
    body: dict | None = None,
    query: dict | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    _require_configured()
    url = settings.heygen.base_url.rstrip("/") + path
    if query:
        url = f"{url}?{urlencode({k: v for k, v in query.items() if v is not None})}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(url, data=data, method=method, headers=_headers(json_body=body is not None))
    try:
        with urlopen(req, timeout=timeout or settings.heygen.request_timeout) as resp:
            raw = resp.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000] if exc.fp else ""
        raise HeyGenError(
            f"HeyGen {method} {path} HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise HeyGenError(f"HeyGen {method} {path} network error: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HeyGenError(f"HeyGen returned non-JSON: {raw[:500]!r}") from exc


# ----------------------------- payload builder ----------------------------- #


def _build_simple_payload(req: HeyGenGenerateSimpleRequest) -> dict[str, Any]:
    avatar_id = req.avatar_id or settings.heygen.default_avatar_id
    voice_id = req.voice_id or settings.heygen.default_voice_id

    if not avatar_id and not req.talking_photo_id:
        raise HeyGenError(
            "No avatar_id provided and HEYGEN_DEFAULT_AVATAR_ID is empty. "
            "Use /heygen/avatars to discover IDs."
        )

    # character
    character: dict[str, Any] = {
        "type": "talking_photo" if req.talking_photo_id else "avatar",
    }
    if req.talking_photo_id:
        character["talking_photo_id"] = req.talking_photo_id
    else:
        character["avatar_id"] = avatar_id
    if req.avatar_style:
        character["avatar_style"] = req.avatar_style
    if req.scale is not None:
        character["scale"] = req.scale
    if req.offset_x is not None or req.offset_y is not None:
        character["offset"] = {
            "x": req.offset_x or 0.0,
            "y": req.offset_y or 0.0,
        }
    if req.matting is not None:
        character["matting"] = req.matting

    # voice
    voice: dict[str, Any] = {}
    if req.audio_url or req.audio_asset_id:
        voice["type"] = "audio"
        if req.audio_url:
            voice["audio_url"] = req.audio_url
        if req.audio_asset_id:
            voice["audio_asset_id"] = req.audio_asset_id
    else:
        voice["type"] = "text"
        voice["input_text"] = req.question_text
        if not voice_id:
            raise HeyGenError(
                "No voice_id provided and HEYGEN_DEFAULT_VOICE_ID is empty. "
                "Use /heygen/voices to discover IDs, or provide audio_url instead."
            )
        voice["voice_id"] = voice_id
        if req.speed is not None:
            voice["speed"] = req.speed
        if req.pitch is not None:
            voice["pitch"] = req.pitch
        if req.emotion:
            voice["emotion"] = req.emotion
        if req.locale:
            voice["locale"] = req.locale

    # background
    background: dict[str, Any] | None = None
    if req.background_color:
        background = {"type": "color", "value": req.background_color}
    elif req.background_image_url:
        background = {"type": "image", "url": req.background_image_url}
    elif req.background_video_url:
        background = {"type": "video", "url": req.background_video_url}

    video_input: dict[str, Any] = {"character": character, "voice": voice}
    if background:
        video_input["background"] = background

    payload: dict[str, Any] = {
        "video_inputs": [video_input],
        "dimension": {
            "width": req.width or settings.heygen.default_width,
            "height": req.height or settings.heygen.default_height,
        },
    }
    if req.title:
        payload["title"] = req.title
    if req.caption is not None:
        payload["caption"] = req.caption
    if req.callback_url:
        payload["callback_url"] = req.callback_url
    if req.folder_id:
        payload["folder_id"] = req.folder_id
    return payload


# ----------------------------- public API ----------------------------- #


def _extract_video_id(resp: dict[str, Any]) -> str:
    data = resp.get("data") or {}
    vid = data.get("video_id") or resp.get("video_id")
    if not vid:
        raise HeyGenError(f"HeyGen response missing video_id: {resp}")
    return str(vid)


def _normalize_status(status_resp: dict[str, Any], *, video_id: str) -> HeyGenGenerateResponse:
    data = status_resp.get("data") or status_resp
    status = (data.get("status") or "unknown").lower()
    return HeyGenGenerateResponse(
        video_id=video_id,
        status=status,
        video_url=data.get("video_url") or data.get("video_url_caption"),
        thumbnail_url=data.get("thumbnail_url"),
        duration_seconds=data.get("duration"),
        error=(data.get("error") or {}).get("message") if isinstance(data.get("error"), dict) else data.get("error"),
        raw=status_resp,
    )


_UPLOAD_BASE = "https://upload.heygen.com"

_CONTENT_TYPE_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _upload_talking_photo_bytes(data: bytes, content_type: str) -> dict[str, Any]:
    """POST raw image bytes to HeyGen's talking_photo upload endpoint."""
    _require_configured()
    if content_type not in _CONTENT_TYPE_BY_EXT.values():
        raise HeyGenError(
            f"Unsupported image content-type '{content_type}'. "
            f"Allowed: {sorted(set(_CONTENT_TYPE_BY_EXT.values()))}"
        )
    url = _UPLOAD_BASE + "/v1/talking_photo"
    req = Request(
        url,
        data=data,
        method="POST",
        headers={
            "X-Api-Key": settings.heygen.api_key,
            "Content-Type": content_type,
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=settings.heygen.request_timeout) as resp:
            raw = resp.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000] if exc.fp else ""
        raise HeyGenError(f"HeyGen talking_photo upload HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise HeyGenError(f"HeyGen talking_photo upload network error: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HeyGenError(f"HeyGen upload returned non-JSON: {raw[:500]!r}") from exc


class HeyGenService:
    def describe(self) -> dict[str, Any]:
        return {
            "service": "heygen",
            "configured": bool(settings.heygen.api_key),
            "base_url": settings.heygen.base_url,
            "default_avatar_id": settings.heygen.default_avatar_id or None,
            "default_voice_id": settings.heygen.default_voice_id or None,
            "default_dimension": {
                "width": settings.heygen.default_width,
                "height": settings.heygen.default_height,
            },
            "mirror_to_storage": settings.heygen.mirror_to_storage,
        }

    # discovery
    def list_avatars(self) -> dict[str, Any]:
        return _request("GET", "/v2/avatars")

    def list_voices(self) -> dict[str, Any]:
        return _request("GET", "/v2/voices")

    # talking photo (upload your own photo -> talking_photo_id)
    def upload_talking_photo_bytes(
        self, data: bytes, *, content_type: str = "image/jpeg"
    ) -> dict[str, Any]:
        resp = _upload_talking_photo_bytes(data, content_type)
        d = resp.get("data") or {}
        return {
            "talking_photo_id": d.get("talking_photo_id") or d.get("id"),
            "talking_photo_url": d.get("talking_photo_url") or d.get("url"),
            "raw": resp,
        }

    def upload_talking_photo_from_url(self, image_url: str) -> dict[str, Any]:
        ext = Path(image_url.split("?", 1)[0]).suffix.lower()
        content_type = _CONTENT_TYPE_BY_EXT.get(ext, "image/jpeg")
        req = Request(image_url, headers={"User-Agent": f"{settings.app_name}/{settings.version}"})
        try:
            with urlopen(req, timeout=settings.heygen.request_timeout) as resp:
                data = resp.read()
                remote_ct = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if remote_ct in _CONTENT_TYPE_BY_EXT.values():
                    content_type = remote_ct
        except (HTTPError, URLError) as exc:
            raise HeyGenError(f"failed to fetch image_url: {exc}") from exc
        return self.upload_talking_photo_bytes(data, content_type=content_type)

    # core
    def submit_raw(self, payload: dict[str, Any]) -> str:
        logger.info("heygen submit_raw: inputs=%d", len(payload.get("video_inputs") or []))
        resp = _request("POST", "/v2/video/generate", body=payload)
        return _extract_video_id(resp)

    def submit_simple(self, req: HeyGenGenerateSimpleRequest) -> str:
        return self.submit_raw(_build_simple_payload(req))

    def get_status(self, video_id: str) -> HeyGenGenerateResponse:
        resp = _request(
            "GET",
            "/v1/video_status.get",
            query={"video_id": video_id},
        )
        return _normalize_status(resp, video_id=video_id)

    def wait_for_completion(
        self,
        video_id: str,
        *,
        max_wait_seconds: int | None = None,
        poll_interval_sec: float | None = None,
    ) -> HeyGenGenerateResponse:
        deadline = time.monotonic() + float(
            max_wait_seconds if max_wait_seconds is not None else settings.heygen.poll_max_wait_sec
        )
        interval = float(poll_interval_sec or settings.heygen.poll_interval_sec)
        last: HeyGenGenerateResponse | None = None
        while True:
            last = self.get_status(video_id)
            if last.status in {"completed", "done", "success"}:
                last.status = "completed"
                return last
            if last.status in {"failed", "error"}:
                last.status = "failed"
                return last
            if time.monotonic() >= deadline:
                raise HeyGenError(
                    f"HeyGen video {video_id} not ready within "
                    f"{max_wait_seconds or settings.heygen.poll_max_wait_sec}s "
                    f"(last status={last.status})"
                )
            time.sleep(interval)

    def mirror_video_to_storage(
        self, final: HeyGenGenerateResponse
    ) -> HeyGenGenerateResponse:
        if not final.video_url:
            return final
        jid = new_job_id()
        work = job_dir(jid)
        try:
            local = work / f"{final.video_id}.mp4"
            download_to_file(final.video_url, local)
            final.storage_url = storage_client.upload(
                local, f"heygen/{final.video_id}.mp4", content_type="video/mp4"
            )
            return final
        finally:
            try:
                import shutil
                shutil.rmtree(work, ignore_errors=True)
            except Exception:  # noqa: BLE001
                pass

    # one-shot orchestrators
    def generate_simple(
        self, req: HeyGenGenerateSimpleRequest
    ) -> HeyGenGenerateResponse:
        video_id = self.submit_simple(req)
        if not req.wait_for_completion:
            return HeyGenGenerateResponse(video_id=video_id, status="processing")
        final = self.wait_for_completion(
            video_id, max_wait_seconds=req.max_wait_seconds
        )
        mirror = (
            req.mirror_to_storage
            if req.mirror_to_storage is not None
            else settings.heygen.mirror_to_storage
        )
        if mirror and final.status == "completed":
            final = self.mirror_video_to_storage(final)
        return final

    def generate_raw(
        self,
        req: HeyGenGenerateRawRequest,
        *,
        wait: bool = True,
        max_wait_seconds: int | None = None,
        mirror: bool | None = None,
    ) -> HeyGenGenerateResponse:
        video_id = self.submit_raw(req.model_dump(exclude_none=True))
        if not wait:
            return HeyGenGenerateResponse(video_id=video_id, status="processing")
        final = self.wait_for_completion(video_id, max_wait_seconds=max_wait_seconds)
        do_mirror = mirror if mirror is not None else settings.heygen.mirror_to_storage
        if do_mirror and final.status == "completed":
            final = self.mirror_video_to_storage(final)
        return final


heygen_service = HeyGenService()
