Python Client API
-----------------

.. currentmodule:: mpl_fastapi.remote


.. automodule:: mpl_fastapi.remote
   :no-members:
   :no-undoc-members:



Qt Thin Client
++++++++++++++

.. currentmodule:: mpl_fastapi.remote.backend_qtremote


.. automodule:: mpl_fastapi.remote.backend_qtremote
   :no-members:
   :no-undoc-members:

**Helper functions**


.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   open_launcher
   run_qt_app
   open_remote_figures
   open_remote_figure


**Classes**

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   TransportThread
   FigureCanvasQTRemote
   NavigationToolbar2QTRemote
   SchemaFormBuilder
   UpdateParametersWidget
   FigureLauncherWindow



Generic Thin Client
+++++++++++++++++++

.. currentmodule:: mpl_fastapi.remote.backend_remote


.. automodule:: mpl_fastapi.remote.backend_remote
   :no-members:
   :no-undoc-members:

**helper functions**

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   list_remote_figures

**Classes**

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   FigureCanvasRemote
   RemoteNavigationToolbar2
   RemotePlotInfo
   FigureManagerRemote


Core WS protocol
++++++++++++++++


.. currentmodule:: mpl_fastapi.ws_client


.. automodule:: mpl_fastapi.ws_client
   :no-members:
   :no-undoc-members:


.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   MatplotlibWebSocketClient
   create_httpx_adapter
   create_fastapi_test_client_adapter



.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   WebSocketAdapter
   ContextManagerWebSocketAdapter
   ImageTypeMode
   ImageFormat
   BinaryImageHeader
   parse_binary_image
