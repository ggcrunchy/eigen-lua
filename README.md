# eigen-lua
A wrapper around parts of the Eigen numerical library

In particular, this binding targets [Eigen 3.3](http://eigen.tuxfamily.org/index.php?title=Main_Page). It is
_very_ much a work in progress, though the (incomplete) [docs](https://ggcrunchy.github.io/corona-plugin-docs/DOCS/eigen/api.html)
give some idea of the direction it's taking. Many solvers are mostly fleshed out, with views, block expressions,
and the like being still underway.

Currently I've only tackled Windows. This project is meant first and foremost to be a [Corona SDK](https://coronalabs.com)
plugin, however, so several more platforms should be forthcoming.

That said, with a little effort the relevant bits could be carved out of the support submodules and retargeted
at other platforms.

**Eigen::Map** seems to significantly complicate things (ramping up compilation times; to add insult to injury,
compilation errors seem to wait about twenty minutes to show up!), so there are separate projects for most scalar
types, along with compile-time options to include maps and strided maps. At the moment, boolean matrices are part
of the "core" module, though maybe this will change now that they're fairly fleshed out in their own right.

A small ["CacheStack" module](https://gist.github.com/ggcrunchy/1a653cc1e4555baba7991bfa06bb6a85) goes with this,
with the aim of recycling Lua-side object wrappers in heavy-duty scenarios where lots of temporaries are made.

For now, development is being done elsewhere, but the idea is to snapshot it here every now and then.