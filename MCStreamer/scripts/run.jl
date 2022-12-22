# This is a hack to prevent version errors in some environments
using PyCall
pyimport("h5py")

using MCStreamer

MCStreamer.main()
