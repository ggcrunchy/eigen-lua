# eigen-lua
A wrapper around parts of the Eigen numerical library

In particular, this binding targets [Eigen 3.3](http://eigen.tuxfamily.org/index.php?title=Main_Page). It is
_very_ much a work in progress, though the (incomplete) [docs](https://ggcrunchy.github.io/corona-plugin-docs/DOCS/eigen/api.html)
give some idea of the direction it's taking. Many solvers are mostly fleshed out, views are still being worked on,
and various block expressions and the like are probably up next. A few utilities for loading matrices from memory and
mapping memory blobs are in the idea phase.

Currently I've only tackled Windows. This project is meant first and foremost to be a [Corona SDK](https://coronalabs.com)
plugin, however, so several more platforms should be forthcoming.

That said, with a little effort the support libraries could be swapped out on other platforms. (This is what the
submodules provide.)

**Eigen::Map** seems to significantly complicate things (ramping up compilation times and then, to add insult to
injury, failing at the very end when one has multiple types), so there are separate projects for "all types, no
maps" along with individual types + maps (actually, this seems to fail with an ICE, at least on the VS2013), along
with another one for the "core" interface.

A small ["cachestack" module](https://gist.github.com/ggcrunchy/1a653cc1e4555baba7991bfa06bb6a85) goes with it,
with the aim of recycling matrices in heavy-duty scenarios where a lot of temporaries are made.

For now, development is being done elsewhere, but the idea is to snapshot it here every now and then.