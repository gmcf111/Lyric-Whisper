"""HuggingFace 下载网络配置。

某些网络环境（企业代理 / 安全软件）会对 HTTPS 做 TLS 拦截，
使用本地/企业根 CA 签发证书。Python 默认用 certifi 的 CA 包，
不包含这些本地 CA，会导致所有 HTTPS 请求报
CERTIFICATE_VERIFY_FAILED。

解决方案：用 truststore 让 httpx 使用操作系统证书库
（Windows 系统证书库已信任这些本地 CA），从而验证通过。

此修复对 huggingface_hub 的所有下载（Whisper / Demucs 等，
均通过 get_session()）统一生效。
"""
import ssl

_CONFIGURED = False


def configure_hf_session() -> None:
    """让 huggingface_hub 的 httpx 客户端使用系统证书库。

    幂等：重复调用只生效一次。失败时静默回退到默认行为，
    不影响在证书正常的环境下使用。
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    try:
        import truststore
        import httpx
        from huggingface_hub.utils._http import (
            hf_request_event_hook,
            set_client_factory,
        )

        def _client_factory() -> "httpx.Client":
            ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            return httpx.Client(
                verify=ctx,
                event_hooks={"request": [hf_request_event_hook]},
                follow_redirects=True,
                timeout=None,
            )

        set_client_factory(_client_factory)
        _CONFIGURED = True
    except Exception:
        # 任意环节失败都不应阻断程序启动，回退默认证书行为
        pass


def make_hf_client():
    """创建独立的 httpx.Client，使用系统证书库 (truststore)。

    每个下载 worker 应使用自己的 client，这样在取消下载时可以
    直接 close() 该 client，立即中断阻塞中的网络读取。
    如果使用共享的 get_session()，关闭它会破坏所有后续 HF 操作。

    返回的 client 默认 follow_redirects=True，超时为 30s。
    """
    import httpx

    verify = True
    try:
        import truststore

        verify = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        pass  # truststore 不可用时回退到 httpx 默认证书

    return httpx.Client(
        verify=verify,
        follow_redirects=True,
        timeout=httpx.Timeout(connect=15, read=30, write=15, pool=15),
    )
