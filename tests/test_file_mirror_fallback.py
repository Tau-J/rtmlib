"""Tests for the automatic Hugging Face mirror fallback in
`rtmlib/tools/file.py`.

Context: all openmmlab-hosted checkpoints referenced by rtmlib have been
mirrored to https://huggingface.co/Tau-J/RTMPose (same relative path,
rooted at the part of the URL after `.../mmpose/v1/projects/`). When the
primary `download.openmmlab.com` download fails for any reason, rtmlib
should automatically retry from that mirror instead of failing outright.
"""
import zipfile

import pytest

from rtmlib.tools import file as file_mod
from rtmlib.tools.file import get_mirror_url

OPENMMLAB_PREFIX = 'https://download.openmmlab.com/mmpose/v1/projects/'
HF_MIRROR_PREFIX = 'https://huggingface.co/Tau-J/RTMPose/resolve/main/'


class TestGetMirrorUrl:

    def test_known_prefix_is_mirrored(self):
        url = OPENMMLAB_PREFIX + 'rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip'
        assert get_mirror_url(url) == HF_MIRROR_PREFIX + 'rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip'

    def test_rtmo_and_rtmw_prefixes_are_also_mirrored(self):
        for sub_project in ('rtmo', 'rtmw', 'rtmposev1'):
            url = OPENMMLAB_PREFIX + f'{sub_project}/onnx_sdk/some_model.zip'
            assert get_mirror_url(url) == HF_MIRROR_PREFIX + f'{sub_project}/onnx_sdk/some_model.zip'

    def test_unrelated_host_has_no_mirror(self):
        url = 'https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-b-coco.onnx'
        assert get_mirror_url(url) is None

    def test_github_releases_have_no_mirror(self):
        url = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx'
        assert get_mirror_url(url) is None


class TestDownloadCheckpointFallback:

    def test_falls_back_to_mirror_on_primary_failure(self, monkeypatch, tmp_path):
        calls = []

        def fake_download_url_to_file(url, dst, hash_prefix=None, progress=True, timeout=30):
            calls.append(url)
            if 'download.openmmlab.com' in url:
                raise ConnectionError('primary host unreachable')
            # Simulate a successful mirror download by writing a minimal
            # zip file containing a single `end2end.onnx` entry, matching
            # what `download_checkpoint` expects to find and rename.
            with zipfile.ZipFile(dst, 'w') as zf:
                zf.writestr('end2end.onnx', b'fake-onnx-bytes')

        monkeypatch.setattr(file_mod, 'download_url_to_file', fake_download_url_to_file)

        url = OPENMMLAB_PREFIX + 'rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip'
        result = file_mod.download_checkpoint(url, dst_dir=str(tmp_path))

        assert len(calls) == 2
        assert calls[0] == url
        assert calls[1] == get_mirror_url(url)
        assert result.endswith('.onnx')
        with open(result, 'rb') as f:
            assert f.read() == b'fake-onnx-bytes'

    def test_no_fallback_available_reraises_original_error(self, monkeypatch, tmp_path):

        def fake_download_url_to_file(url, dst, hash_prefix=None, progress=True, timeout=30):
            raise ConnectionError('host unreachable, and no mirror exists')

        monkeypatch.setattr(file_mod, 'download_url_to_file', fake_download_url_to_file)

        # A host with no registered mirror.
        url = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx'
        with pytest.raises(ConnectionError):
            file_mod.download_checkpoint(url, dst_dir=str(tmp_path))

    def test_primary_success_never_touches_mirror(self, monkeypatch, tmp_path):
        calls = []

        def fake_download_url_to_file(url, dst, hash_prefix=None, progress=True, timeout=30):
            calls.append(url)
            with open(dst, 'wb') as f:
                f.write(b'fake-onnx-bytes')

        monkeypatch.setattr(file_mod, 'download_url_to_file', fake_download_url_to_file)

        url = OPENMMLAB_PREFIX + 'rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.onnx'
        file_mod.download_checkpoint(url, dst_dir=str(tmp_path))

        assert calls == [url]
