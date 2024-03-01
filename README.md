# Package to preserve decoding test studies at Fermilab

The [stim playground notebook](https://github.com/usarica/FNAL-QCDecodingTests/blob/master/Stim_playground.ipynb) introduces a few different wrapper functions I wrote to make the collection of statistics and plotting easier.

The [notebook for distance-4 surface codes](https://github.com/usarica/FNAL-QCDecodingTests/blob/master/surface_code_d4.ipynb) outlines my progress on initial studies for a simple surface code over which one could run different decoding algorithms.
- In this notebook, I have been able to run pyMatching successfully.
- It looks like running a [stim-based BP+OSD implementation](https://github.com/oscarhiggott/stimbposd) takes a very long time to run for a very small number of shots on my personal computer, so I stopped that for now.
- Let's see how we can interface ML decoders.

In both notebooks, I assume the probabilities of all noise channels are the same, which, I suppose, does not have to be the case.
