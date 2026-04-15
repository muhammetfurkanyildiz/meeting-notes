"""
droplet_manager.py
──────────────────
Digital Ocean droplet yönetimi.

Kullanım:
  python devops/droplet_manager.py create
  python devops/droplet_manager.py destroy
  python devops/droplet_manager.py status
  python devops/droplet_manager.py deploy
  python devops/droplet_manager.py update
  python devops/droplet_manager.py logs

Gerekli .env değişkenleri:
  DO_API_TOKEN      — Digital Ocean kişisel erişim token'ı
  DO_SSH_KEY_ID     — DO hesabındaki SSH key ID (sayı)
  SSH_KEY_PATH      — Yerel özel anahtar yolu (~/.ssh/id_rsa)
  DROPLET_NAME      — Droplet adı (varsayılan: meeting-notes)
  DROPLET_REGION    — Bölge slug (varsayılan: fra1)
  DROPLET_SIZE      — Boyut slug (varsayılan: s-2vcpu-4gb)
  GIT_REPO_URL      — https://github.com/kullanici/meeting-notes.git
  GIT_BRANCH        — main
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# paramiko isteğe bağlı; SSH adımlarında import edilir
try:
    import paramiko
    _PARAMIKO_OK = True
except ImportError:
    _PARAMIKO_OK = False

# ── .env yükle ────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

# ── Renkli terminal çıktısı ───────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    GREEN  = "\033[32m"
    RED    = "\033[31m"
    YELLOW = "\033[33m"
    BLUE   = "\033[34m"
    BOLD   = "\033[1m"

def ok(msg: str)    -> None: print(f"{C.GREEN}{C.BOLD}✓{C.RESET}  {msg}")
def err(msg: str)   -> None: print(f"{C.RED}{C.BOLD}✗{C.RESET}  {msg}", file=sys.stderr)
def wait(msg: str)  -> None: print(f"{C.YELLOW}…{C.RESET}  {msg}")
def info(msg: str)  -> None: print(f"{C.BLUE}ℹ{C.RESET}  {msg}")
def bold(msg: str)  -> None: print(f"{C.BOLD}{msg}{C.RESET}")
def die(msg: str)   -> None:
    err(msg)
    sys.exit(1)

# ── Yapılandırma ──────────────────────────────────────────────────────────────

def _require(key: str) -> str:
    val = os.getenv(key, "").strip()
    if not val:
        die(f"Ortam değişkeni eksik: {key}  (.env dosyasını kontrol edin)")
    return val

DO_API_TOKEN  = lambda: _require("DO_API_TOKEN")
DO_SSH_KEY_ID = lambda: _require("DO_SSH_KEY_ID")
SSH_KEY_PATH  = lambda: os.getenv("SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_rsa"))
DROPLET_NAME  = lambda: os.getenv("DROPLET_NAME", "meeting-notes")
DROPLET_REGION= lambda: os.getenv("DROPLET_REGION", "fra1")
DROPLET_SIZE  = lambda: os.getenv("DROPLET_SIZE", "s-2vcpu-4gb")
GIT_REPO_URL  = lambda: _require("GIT_REPO_URL")
GIT_BRANCH    = lambda: os.getenv("GIT_BRANCH", "main")
APP_PORT      = lambda: int(os.getenv("API_PORT", "8000"))

# ── Digital Ocean API ─────────────────────────────────────────────────────────

DO_BASE = "https://api.digitalocean.com/v2"

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {DO_API_TOKEN()}",
        "Content-Type": "application/json",
    }

def _get(path: str) -> dict:
    r = requests.get(f"{DO_BASE}{path}", headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()

def _post(path: str, body: dict) -> dict:
    r = requests.post(f"{DO_BASE}{path}", headers=_headers(), json=body, timeout=30)
    r.raise_for_status()
    return r.json()

def _delete(path: str) -> None:
    r = requests.delete(f"{DO_BASE}{path}", headers=_headers(), timeout=30)
    if r.status_code not in (200, 204):
        r.raise_for_status()

# ── Droplet yardımcıları ──────────────────────────────────────────────────────

def _find_available_region(size_slug: str, poll_interval: int = 60) -> str:
    """
    Verilen boyut için DO API'yi periyodik olarak sorgular;
    müsait bir bölge bulunana kadar poll_interval saniyede bir tekrar eder.
    """
    attempt = 0
    while True:
        attempt += 1
        wait(f"'{size_slug}' için müsait bölgeler taranıyor… (deneme {attempt})")
        try:
            data = _get("/sizes?per_page=200")
        except requests.HTTPError as exc:
            info(f"Boyut listesi alınamadı, tekrar denenecek: {exc}")
            time.sleep(poll_interval)
            continue

        for s in data.get("sizes", []):
            if s["slug"] == size_slug:
                regions = s.get("regions", [])
                if regions:
                    info(f"Müsait bölgeler: {', '.join(regions)}")
                    return regions[0]
                break  # slug bulundu ama müsait bölge yok

        info(f"Henüz müsait bölge yok. {poll_interval}s sonra tekrar denenecek… (Ctrl+C ile iptal)")
        time.sleep(poll_interval)


def _find_droplet() -> dict | None:
    """İsme göre mevcut droplet'i bulur; yoksa None döner."""
    name = DROPLET_NAME()
    data = _get("/droplets")
    for d in data.get("droplets", []):
        if d["name"] == name:
            return d
    return None

def _droplet_ip(droplet: dict) -> str | None:
    """Droplet'in public IPv4 adresini döner."""
    for net in droplet.get("networks", {}).get("v4", []):
        if net["type"] == "public":
            return net["ip_address"]
    return None

def _wait_active(droplet_id: int, timeout: int = 600) -> dict:
    """
    Droplet 'active' durumuna geçene kadar bekler.
    timeout: maksimum bekleme süresi (saniye), varsayılan 10 dk.
    """
    start = time.time()
    while True:
        elapsed = int(time.time() - start)
        if elapsed > timeout:
            die(f"Zaman aşımı: droplet {timeout}s içinde aktif olmadı.")
        d = _get(f"/droplets/{droplet_id}")["droplet"]
        status = d["status"]
        ip     = _droplet_ip(d)
        if status == "active" and ip:
            return d
        wait(f"Droplet durumu: {status} — {elapsed}s beklendi, 30s sonra tekrar kontrol…")
        time.sleep(30)

# ── SSH / SFTP yardımcıları ───────────────────────────────────────────────────

def _ssh_client(ip: str) -> "paramiko.SSHClient":
    """SSH bağlantısı kurar; bağlantı hazır olana kadar retry yapar (max 5 dk)."""
    if not _PARAMIKO_OK:
        die("paramiko yüklü değil: pip install paramiko")

    key_path = SSH_KEY_PATH()
    if not Path(key_path).exists():
        die(f"SSH anahtarı bulunamadı: {key_path}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        pkey = paramiko.Ed25519Key.from_private_key_file(key_path)
    except Exception:
        try:
            pkey = paramiko.RSAKey.from_private_key_file(key_path)
        except Exception:
            pkey = paramiko.ECDSAKey.from_private_key_file(key_path)

    start = time.time()
    while True:
        try:
            client.connect(ip, username="root", pkey=pkey, timeout=10)
            return client
        except Exception as exc:
            if time.time() - start > 300:
                die(f"SSH bağlantısı kurulamadı ({ip}): {exc}")
            wait(f"SSH henüz hazır değil — 15s sonra tekrar denenecek… ({exc})")
            time.sleep(15)


def _ssh_run(client: "paramiko.SSHClient", cmd: str, stream: bool = False) -> str:
    """
    SSH üzerinden komut çalıştırır.
    stream=True ise çıktı anlık yazdırılır (uzun kurulum adımları için).
    Hata durumunda die() çağrılır.
    """
    _, stdout, stderr = client.exec_command(cmd, get_pty=stream)

    if stream:
        for line in iter(stdout.readline, ""):
            print(f"    {line}", end="")
        exit_code = stdout.channel.recv_exit_status()
    else:
        exit_code = stdout.channel.recv_exit_status()

    if exit_code != 0:
        error_out = stderr.read().decode(errors="replace").strip()
        die(f"Komut başarısız (exit {exit_code}): {cmd}\n    {error_out}")

    return stdout.read().decode(errors="replace") if not stream else ""


def _sftp_send(client: "paramiko.SSHClient", local_path: str, remote_path: str) -> None:
    """Yerel dosyayı SFTP ile uzak sunucuya gönderir."""
    if not Path(local_path).exists():
        die(f"Gönderilecek dosya bulunamadı: {local_path}")
    sftp = client.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()


# ── Komutlar ──────────────────────────────────────────────────────────────────

def cmd_create() -> dict:
    """Yeni bir droplet oluşturur ve aktif olana kadar bekler."""
    bold("\n── Droplet Oluşturuluyor ──────────────────────────────")

    existing = _find_droplet()
    if existing:
        ip = _droplet_ip(existing)
        info(f"'{DROPLET_NAME()}' adında droplet zaten mevcut → {ip}")
        return existing

    size = DROPLET_SIZE()

    # .env'de bölge belirtilmemişse veya "auto" ise otomatik tara
    region = DROPLET_REGION()
    if not region or region == "auto":
        region = _find_available_region(size)
        ok(f"Otomatik seçilen bölge: {C.BOLD}{region}{C.RESET}")
    else:
        info(f"Bölge: {region}  |  Boyut: {size}  |  Ad: {DROPLET_NAME()}")

    wait("Droplet oluşturuluyor…")

    try:
        resp = _post("/droplets", {
            "name":     DROPLET_NAME(),
            "region":   region,
            "size":     size,
            "image":    "ubuntu-22-04-x64",
            "ssh_keys": [DO_SSH_KEY_ID()],
            "backups":  False,
            "ipv6":     False,
            "tags":     ["meeting-notes"],
        })
    except requests.HTTPError as exc:
        die(f"Droplet oluşturulamadı: {exc.response.text}")

    droplet_id = resp["droplet"]["id"]
    info(f"Droplet ID: {droplet_id} — aktif olana kadar bekleniyor…")
    droplet = _wait_active(droplet_id)

    ip = _droplet_ip(droplet)
    ok(f"Droplet aktif!")
    info(f"IP Adresi  : {C.BOLD}{ip}{C.RESET}")
    info(f"Droplet ID : {droplet_id}")
    return droplet


def cmd_destroy() -> None:
    """Droplet'i durdurur ve siler."""
    bold("\n── Droplet Siliniyor ──────────────────────────────────")

    droplet = _find_droplet()
    if not droplet:
        info(f"'{DROPLET_NAME()}' adında aktif droplet bulunamadı.")
        return

    ip = _droplet_ip(droplet)
    info(f"Droplet: {droplet['id']}  |  IP: {ip}")

    # Servisi durdur (SSH bağlanabiliyorsa)
    if ip and _PARAMIKO_OK:
        try:
            wait("Servis durduruluyor (systemctl stop meeting-notes)…")
            client = _ssh_client(ip)
            _ssh_run(client, "systemctl stop meeting-notes || true")
            client.close()
            ok("Servis durduruldu.")
        except Exception as exc:
            info(f"Servis durdurma atlandı (SSH hatası): {exc}")

    wait(f"Droplet siliniyor (ID: {droplet['id']})…")
    try:
        _delete(f"/droplets/{droplet['id']}")
    except requests.HTTPError as exc:
        die(f"Droplet silinemedi: {exc.response.text}")

    ok("Droplet silindi.")
    ok(f"{C.BOLD}Ücretlendirme durdu.{C.RESET}")


def cmd_status() -> None:
    """Droplet durumunu gösterir."""
    bold("\n── Droplet Durumu ─────────────────────────────────────")

    droplet = _find_droplet()
    if not droplet:
        info(f"'{DROPLET_NAME()}' adında droplet bulunamadı.")
        return

    ip      = _droplet_ip(droplet) or "—"
    status  = droplet["status"]
    created = droplet.get("created_at", "—")
    size    = droplet.get("size", {})
    cost    = size.get("price_monthly", "?")
    vcpus   = size.get("vcpus", "?")
    ram_mb  = size.get("memory", "?")

    status_color = C.GREEN if status == "active" else C.YELLOW

    print()
    print(f"  Ad         : {C.BOLD}{droplet['name']}{C.RESET}")
    print(f"  Durum      : {status_color}{C.BOLD}{status}{C.RESET}")
    print(f"  IP Adresi  : {C.BOLD}{ip}{C.RESET}")
    print(f"  vCPU / RAM : {vcpus} vCPU / {ram_mb} MB")
    print(f"  Aylık Maliyet : ${cost}")
    print(f"  Oluşturulma   : {created}")
    print(f"  Uygulama URL  : http://{ip}:{APP_PORT()}")
    print()


def cmd_deploy() -> None:
    """
    Sıfırdan tam kurulum yapar:
    create → SSH bekle → paketler → git clone → .env → venv → systemd
    """
    bold("\n── Tam Kurulum (Deploy) ───────────────────────────────")

    droplet = cmd_create()
    ip      = _droplet_ip(droplet)
    if not ip:
        die("Droplet IP adresi alınamadı.")

    info(f"Sunucu IP: {ip}")
    wait("SSH bağlantısı bekleniyor…")

    try:
        client = _ssh_client(ip)
        ok("SSH bağlantısı kuruldu.")

        # ── 1. Sistem paketleri ───────────────────────────────────────────────
        wait("Cloud-init ve apt kilidi bekleniyor…")
        _ssh_run(client, "cloud-init status --wait || true")
        _ssh_run(client, "while fuser /var/lib/apt/lists/lock /var/lib/dpkg/lock /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do echo 'apt kilidi bekleniyor...'; sleep 5; done")
        wait("Sistem paketleri güncelleniyor (bu 2-3 dakika sürebilir)…")
        _ssh_run(client, "DEBIAN_FRONTEND=noninteractive apt-get update -y", stream=True)
        _ssh_run(
            client,
            "DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "python3.11 python3.11-venv python3-pip ffmpeg git "
            "libpango-1.0-0 libpangoft2-1.0-0 libpangocairo-1.0-0 "
            "libgdk-pixbuf2.0-0 libffi-dev shared-mime-info",
            stream=True,
        )
        ok("Sistem paketleri kuruldu.")

        # ── 2. Git clone ──────────────────────────────────────────────────────
        wait(f"Depo klonlanıyor: {GIT_REPO_URL()}")
        _ssh_run(client, "rm -rf /opt/meeting-notes")
        _ssh_run(client, f"git clone {GIT_REPO_URL()} /opt/meeting-notes", stream=True)
        _ssh_run(client, f"cd /opt/meeting-notes && git checkout {GIT_BRANCH()}")
        ok(f"Depo klonlandı (branch: {GIT_BRANCH()}).")

        # ── 3. .env dosyasını gönder ──────────────────────────────────────────
        local_env = str(_ROOT / ".env")
        wait(".env dosyası sunucuya gönderiliyor…")
        _sftp_send(client, local_env, "/opt/meeting-notes/.env")
        ok(".env dosyası gönderildi.")

        # ── 4. Virtual environment + bağımlılıklar ────────────────────────────
        wait("Python sanal ortamı oluşturuluyor…")
        _ssh_run(client, "cd /opt/meeting-notes && python3.11 -m venv venv")
        _ssh_run(client, "cd /opt/meeting-notes && venv/bin/pip install --upgrade pip", stream=True)
        wait("Python bağımlılıkları yükleniyor (bu uzun sürebilir)…")
        _ssh_run(client, "cd /opt/meeting-notes && venv/bin/pip install -r requirements.txt", stream=True)
        ok("Python ortamı hazır.")

        # ── 5. systemd service ────────────────────────────────────────────────
        wait("systemd servisi oluşturuluyor…")
        service_content = _SYSTEMD_SERVICE.replace("\\n", "\n")
        # Uzak dosyaya yaz
        _ssh_run(
            client,
            f"cat > /etc/systemd/system/meeting-notes.service << 'SVCEOF'\n{service_content}\nSVCEOF",
        )
        _ssh_run(client, "systemctl daemon-reload")
        _ssh_run(client, "systemctl enable meeting-notes")
        _ssh_run(client, "systemctl start meeting-notes")
        ok("Servis başlatıldı.")

        # ── 6. Doğrulama ──────────────────────────────────────────────────────
        time.sleep(5)  # servisin ayağa kalkması için kısa bekle
        wait("Servis durumu kontrol ediliyor…")
        _, out, _ = client.exec_command("systemctl is-active meeting-notes")
        active = out.read().decode().strip()

        if active == "active":
            ok("Servis çalışıyor.")
        else:
            _, jout, _ = client.exec_command(
                "journalctl -u meeting-notes -n 30 --no-pager"
            )
            logs = jout.read().decode(errors="replace")
            err(f"Servis başlatılamadı (durum: {active}). Son loglar:\n{logs}")

        client.close()

        bold("\n" + "─" * 55)
        ok(f"{C.BOLD}Deploy tamamlandı!{C.RESET}")
        info(f"Uygulama adresi : {C.BOLD}http://{ip}:{APP_PORT()}{C.RESET}")
        info(f"API Docs        : http://{ip}:{APP_PORT()}/docs")
        bold("─" * 55 + "\n")

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        err(f"Deploy başarısız: {exc}")
        wait("Droplet otomatik siliniyor (hatalı kaynak temizleniyor)…")
        try:
            _delete(f"/droplets/{droplet['id']}")
            ok("Droplet silindi.")
        except Exception as del_exc:
            err(f"Droplet silinemedi, manuel silmeniz gerekebilir: {del_exc}")
        sys.exit(1)


def cmd_update() -> None:
    """Mevcut droplet'te git pull + pip install + servis yeniden başlat."""
    bold("\n── Güncelleme ─────────────────────────────────────────")

    droplet = _find_droplet()
    if not droplet:
        die(f"Aktif droplet bulunamadı. Önce 'deploy' komutunu çalıştırın.")

    ip = _droplet_ip(droplet)
    info(f"Sunucu IP: {ip}")
    wait("SSH bağlantısı kuruluyor…")
    client = _ssh_client(ip)
    ok("SSH bağlantısı kuruldu.")

    wait(f"Kod güncelleniyor (git pull origin {GIT_BRANCH()})…")
    _ssh_run(client, f"cd /opt/meeting-notes && git pull origin {GIT_BRANCH()}", stream=True)

    wait("Bağımlılıklar güncelleniyor…")
    _ssh_run(client,
        "cd /opt/meeting-notes && venv/bin/pip install "
        "--upgrade setuptools wheel pip",
        stream=True)
    _ssh_run(client, "cd /opt/meeting-notes && venv/bin/pip install -r requirements.txt", stream=True)

    wait("Servis yeniden başlatılıyor…")
    # Servis yoksa önce kur
    _ssh_run(client, """
if ! systemctl is-enabled meeting-notes > /dev/null 2>&1; then
    cat > /etc/systemd/system/meeting-notes.service << 'EOF'
[Unit]
Description=Meeting Notes AI
After=network.target

[Service]
WorkingDirectory=/opt/meeting-notes
ExecStart=/opt/meeting-notes/venv/bin/python main.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable meeting-notes
fi
systemctl restart meeting-notes
""")

    time.sleep(3)
    _, out, _ = client.exec_command("systemctl is-active meeting-notes")
    active = out.read().decode().strip()

    client.close()

    if active == "active":
        ok(f"Güncelleme tamamlandı!  Servis aktif: http://{ip}:{APP_PORT()}")
    else:
        err(f"Güncelleme yapıldı ancak servis şu anda '{active}' durumunda. Logları kontrol edin.")


def cmd_logs() -> None:
    """Canlı servis loglarını terminale akıtır."""
    bold("\n── Canlı Loglar (Ctrl+C ile çık) ─────────────────────")

    droplet = _find_droplet()
    if not droplet:
        die("Aktif droplet bulunamadı.")

    ip = _droplet_ip(droplet)
    info(f"Sunucu IP: {ip}")
    wait("SSH bağlantısı kuruluyor…")

    if not _PARAMIKO_OK:
        die("paramiko yüklü değil: pip install paramiko")

    client = _ssh_client(ip)
    ok("Bağlantı kuruldu. Loglar akıyor…\n")

    try:
        channel = client.get_transport().open_session()
        channel.get_pty()
        channel.exec_command("journalctl -u meeting-notes -f --no-pager")

        while True:
            if channel.recv_ready():
                data = channel.recv(4096).decode(errors="replace")
                print(data, end="", flush=True)
            if channel.exit_status_ready():
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n")
        info("Log izleme durduruldu.")
    finally:
        client.close()


def cmd_ssh() -> None:
    """Droplet'e interaktif SSH bağlantısı açar."""
    droplet = _find_droplet()
    if not droplet:
        die("Aktif droplet bulunamadı.")
    ip = _droplet_ip(droplet)
    key_path = SSH_KEY_PATH()
    info(f"Sunucuya bağlanılıyor: {ip}")
    import subprocess
    subprocess.run([
        "ssh", "-i", key_path,
        "-o", "StrictHostKeyChecking=no",
        f"root@{ip}"
    ])


# ── systemd service şablonu ───────────────────────────────────────────────────

_SYSTEMD_SERVICE = """[Unit]
Description=Meeting Notes AI
After=network.target

[Service]
WorkingDirectory=/opt/meeting-notes
ExecStart=/opt/meeting-notes/venv/bin/python main.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="droplet_manager.py",
        description="Digital Ocean Meeting Notes Droplet Yöneticisi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Komutlar:
  create   Yeni droplet oluştur
  destroy  Droplet'i durdur ve sil
  status   Droplet durumunu göster
  deploy   Sıfırdan tam kurulum yap
  update   Kodu güncelle, servisi yeniden başlat
  logs     Canlı servis loglarını izle
  ssh      Sunucuya interaktif SSH bağlantısı aç
        """,
    )
    parser.add_argument(
        "command",
        choices=["create", "destroy", "status", "deploy", "update", "logs", "ssh"],
        help="Çalıştırılacak komut",
    )
    args = parser.parse_args()

    commands = {
        "create":  cmd_create,
        "destroy": cmd_destroy,
        "status":  cmd_status,
        "deploy":  cmd_deploy,
        "update":  cmd_update,
        "logs":    cmd_logs,
        "ssh":     cmd_ssh,
    }

    try:
        commands[args.command]()
    except KeyboardInterrupt:
        print()
        info("İşlem kullanıcı tarafından iptal edildi.")
        sys.exit(0)
    except requests.HTTPError as exc:
        die(f"Digital Ocean API hatası: {exc.response.status_code} — {exc.response.text}")
    except requests.ConnectionError:
        die("Ağ bağlantısı kurulamadı. İnternet bağlantınızı kontrol edin.")


if __name__ == "__main__":
    main()
