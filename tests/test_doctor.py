from backend import config, device, doctor
from backend.doctor import FAIL, OK, WARN, CheckResult


def test_parse_version_variants():
    assert doctor.parse_version("ffmpeg version 7.1-full_build-www.gyan.dev") == (7, 1)
    assert doctor.parse_version("ffmpeg version n7.1.1") == (7, 1, 1)
    assert doctor.parse_version("ollama version is 0.12.3") == (0, 12, 3)
    assert doctor.parse_version("no version here") is None
    assert doctor.parse_version("") is None


def test_python_status():
    assert doctor.python_status((3, 9)) == FAIL
    assert doctor.python_status(config.PYTHON_MIN) == OK
    assert doctor.python_status((3, 11)) == OK
    assert doctor.python_status(config.PYTHON_MAX) == OK
    assert doctor.python_status((3, 13)) == WARN


def test_ram_and_disk_status():
    assert doctor.ram_status(4) == FAIL
    assert doctor.ram_status(12) == WARN
    assert doctor.ram_status(16) == OK
    assert doctor.disk_status(2) == FAIL
    assert doctor.disk_status(50) == OK


def test_overall():
    ok = CheckResult("a", OK, "")
    warn = CheckResult("b", WARN, "")
    fail = CheckResult("c", FAIL, "")
    assert doctor.overall([ok]) == OK
    assert doctor.overall([ok, warn]) == WARN
    assert doctor.overall([ok, warn, fail]) == FAIL


def test_install_hint_per_os():
    assert "winget" in doctor.install_hint("ffmpeg", "win32")
    assert "apt" in doctor.install_hint("ffmpeg", "linux")
    assert "brew" in doctor.install_hint("ffmpeg", "darwin")


def test_parse_nvidia_smi():
    info = device.parse_nvidia_smi("NVIDIA GeForce RTX 4050 Laptop GPU, 6141\n")
    assert info.name == "NVIDIA GeForce RTX 4050 Laptop GPU"
    assert info.vram_mb == 6141
    assert device.parse_nvidia_smi("") is None


def test_models_for():
    assert device.models_for("gpu")["asr"] == "large-v3-turbo"
    assert device.models_for("cpu")["asr"] == "medium"
    try:
        device.models_for("tpu")
    except ValueError:
        pass
    else:
        raise AssertionError("harus ValueError")
