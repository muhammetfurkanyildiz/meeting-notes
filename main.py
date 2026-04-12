"""
main.py
───────
Uygulamanın giriş noktası.
`python main.py` komutuyla uvicorn başlatır.
"""

import logging
import uvicorn
import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("main")


def main() -> None:
    logger.info("=" * 55)
    logger.info("  Toplantı Notları Uygulaması başlatılıyor…")
    logger.info("  Adres : http://%s:%d", config.API_HOST, config.API_PORT)
    logger.info("  API   : http://%s:%d/docs", config.API_HOST, config.API_PORT)
    logger.info("  Log   : %s", config.LOG_LEVEL)
    logger.info("=" * 55)

    uvicorn.run(
        "api.server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        reload=False,        # model ağırlıkları yeniden yüklenmemeli
        log_level=config.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
