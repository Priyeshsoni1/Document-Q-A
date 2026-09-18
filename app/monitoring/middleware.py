import time
import uuid

from fastapi import Request

from app.monitoring.logger import logger


async def request_logging_middleware(
    request: Request,
    call_next,
):
    """Log every HTTP request with latency."""

    request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    start_time = time.perf_counter()

    try:

        response = await call_next(request)

        status_code = response.status_code

    except Exception:

        status_code = 500

        raise

    finally:

        duration_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        logger.info(
            "request_id=%s method=%s "
            "path=%s status=%s latency_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            status_code,
            duration_ms,
        )

    response.headers[
        "X-Request-ID"
    ] = request_id

    return response