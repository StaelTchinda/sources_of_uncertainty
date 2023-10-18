
from typing import List, Optional, Text
from matplotlib import pyplot as plt
import numpy as np

def histogram_vector(data: np.ndarray, title: Optional[Text] = None, label: Optional[List[Text]] = None):
    fig, ax = plt.subplots()

    # We can set the number of bins with the *bins* keyword argument.
    if label is not None:
        ax.hist(data, label=label)
    else:
        ax.hist(data)
    if title is not None:
        ax.set_title(title)
    return fig



# """The library seems to have a deep bug. I used this temporary fix to use it."""
from torch.utils.tensorboard.writer import *
# import numpy as np
# from torch.utils.tensorboard.summary import Summary

# import os
# import time
# import torch

# from tensorboard.compat import tf
# from tensorboard.compat.proto.event_pb2 import SessionLog
# from tensorboard.compat.proto.event_pb2 import Event
# from tensorboard.compat.proto import event_pb2
# from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
# from tensorboard.summary.writer.event_file_writer import EventFileWriter

# from torch.utils.tensorboard._convert_np import make_np
# from torch.utils.tensorboard._embedding import (
#     make_mat,
#     make_sprite, 
#     make_tsv,
#     write_pbtxt,
#     get_embedding_info,
# )
# from torch.utils.tensorboard._onnx_graph import load_onnx_graph
# from torch.utils.tensorboard._pytorch_graph import graph
# from torch.utils.tensorboard._utils import figure_to_image
# from torch.utils.tensorboard.summary import (
#     scalar,
#     histogram,
#     histogram_raw,
#     image,
#     audio,
#     text,
#     pr_curve,
#     pr_curve_raw,
#     video,
#     custom_scalars,
#     image_boxes,
#     mesh,
#     hparams,
# )
# def add_histogram(
#         self,
#         tag,
#         values,
#         global_step=None,
#         bins="tensorflow",
#         walltime=None,
#         max_bins=None,
#     ):
#         """Add histogram to summary.

#         Args:
#             tag (str): Data identifier
#             values (torch.Tensor, numpy.ndarray, or string/blobname): Values to build histogram
#             global_step (int): Global step value to record
#             bins (str): One of {'tensorflow','auto', 'fd', ...}. This determines how the bins are made. You can find
#               other options in: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
#             walltime (float): Optional override default walltime (time.time())
#               seconds after epoch of event

#         Examples::

#             from torch.utils.tensorboard import SummaryWriter
#             import numpy as np
#             writer = SummaryWriter()
#             for i in range(10):
#                 x = np.random.random(1000)
#                 writer.add_histogram('distribution centers', x + i, i)
#             writer.close()

#         Expected result:

#         .. image:: _static/img/tensorboard/add_histogram.png
#            :scale: 50 %

#         """
#         torch._C._log_api_usage_once("tensorboard.logging.add_histogram")
#         if self._check_caffe2_blob(values):
#             from caffe2.python import workspace

#             values = workspace.FetchBlob(values)
#         if isinstance(bins, str) and bins == "tensorflow":
#             bins = self.default_bins
#         self._get_file_writer().add_summary(
#             histogram(tag, values, bins, max_bins=max_bins), global_step, walltime
#         )

# def histogram(name, values, bins, max_bins=None):
#     # pylint: disable=line-too-long
#     """Outputs a `Summary` protocol buffer with a histogram.
#     The generated
#     [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
#     has one summary value containing a histogram for `values`.
#     This op reports an `InvalidArgument` error if any value is not finite.
#     Args:
#       name: A name for the generated node. Will also serve as a series name in
#         TensorBoard.
#       values: A real numeric `Tensor`. Any shape. Values to use to
#         build the histogram.
#     Returns:
#       A scalar `Tensor` of type `string`. The serialized `Summary` protocol
#       buffer.
#     """
#     values = make_np(values)
#     hist = make_histogram(values.astype(float), bins, max_bins)
#     return Summary(value=[Summary.Value(tag=name, histo=hist)])
#
# 
# def make_histogram(values, bins, max_bins=None):
#     """Convert values into a histogram proto using logic from histogram.cc."""
#     if values.size == 0:
#         raise ValueError("The input has no element.")
#     values = values.reshape(-1)
#     counts, limits = np.histogram(values, bins=bins)
#     num_bins = len(counts)
#     if max_bins is not None and num_bins > max_bins:
#         subsampling = num_bins // max_bins
#         subsampling_remainder = num_bins % subsampling
#         if subsampling_remainder != 0:
#             counts = np.pad(
#                 counts,
#                 pad_width=[[0, subsampling - subsampling_remainder]],
#                 mode="constant",
#                 constant_values=0,
#             )
#         counts = counts.reshape(-1, subsampling).sum(axis=-1)
#         new_limits = np.empty((counts.size + 1,), limits.dtype)
#         new_limits[:-1] = limits[:-1:subsampling]
#         new_limits[-1] = limits[-1]
#         limits = new_limits

#     # Find the first and the last bin defining the support of the histogram:
#     # cum_counts = np.cumsum(np.greater(counts, 0, dtype=np.int32))
#     cum_counts = np.cumsum(np.greater(counts, 0)) # Error fixed in commit https://github.com/pytorch/pytorch/commit/176d00bd68468be5c559d453ee524d796cbdee00
#     start, end = np.searchsorted(cum_counts, [0, cum_counts[-1] - 1], side="right")
#     start = int(start)
#     end = int(end) + 1
#     del cum_counts

#     # TensorBoard only includes the right bin limits. To still have the leftmost limit
#     # included, we include an empty bin left.
#     # If start == 0, we need to add an empty one left, otherwise we can just include the bin left to the
#     # first nonzero-count bin:
#     counts = (
#         counts[start - 1 : end] if start > 0 else np.concatenate([[0], counts[:end]])
#     )
#     limits = limits[start : end + 1]

#     if counts.size == 0 or limits.size == 0:
#         raise ValueError("The histogram is empty, please file a bug report.")

#     sum_sq = values.dot(values)
#     return HistogramProto(
#         min=values.min(),
#         max=values.max(),
#         num=len(values),
#         sum=values.sum(),
#         sum_squares=sum_sq,
#         bucket_limit=limits.tolist(),
#         bucket=counts.tolist(),
#     )