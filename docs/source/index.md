# mpl-fastapi documentation

The goal of this project is to provide a [fastapi](https://fastapi.tiangolo.com)
[`APIRouter`](https://fastapi.tiangolo.com/reference/apirouter/?h=apirouter) that
serves interactive [Matplotlib](https://matplotlib.org) figures via a websocket.
The provides the ability to serve visualization developed in Python via Matplotlib
to the web with server-side rendering.


```{admonition} 🤖 generated code ahead
This project was half about generating a useful tool and half about having a
real project to try out LLM tools on.  The initial proof of concept was done almost
4 years ago (late 2021/early 2022) and then sat somewhere too far down my todo list
to ever get done until early 2026.

I am not sure about the future of LLMs in this project.  I suspect I'll do a few more
big things, but if this moves beyond being a toy will move to follow Matplotlib's policy.
```

## Reference

```{toctree}
---
maxdepth: 1
---
API.md
python_client.rst
js_api.rst
websocket-protocol

```
