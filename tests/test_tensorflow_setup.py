import tensorflow as tf
import pytest


# def test_gpu_available():
#     """Test that at least one GPU is available."""
#     assert len(tf.config.list_physical_devices('GPU')) > 0, "No GPU detected."


@pytest.mark.skipif(not tf.test.is_built_with_cuda(), reason="TensorFlow is not built with CUDA")
def test_cuda_version():
    """Test that the correct CUDA version is being used."""
    cuda_version = tf.sysconfig.get_build_info()["cuda_version"]
    assert cuda_version == "12.2", f"Expected CUDA version 12.2, but got {cuda_version}"


@pytest.mark.skipif(not tf.test.is_built_with_cuda(), reason="TensorFlow is not built with CUDA")
def test_cudnn_version():
    """Test that the correct cuDNN version is being used."""
    cudnn_version = tf.sysconfig.get_build_info()["cudnn_version"]
    assert cudnn_version == "8", f"Expected cuDNN version 8, but got {cudnn_version}"
