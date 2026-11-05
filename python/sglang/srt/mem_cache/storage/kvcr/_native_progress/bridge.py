"""Actor-local high-level overlay; installed modules/bindings are never patched."""
import hashlib,importlib.util,inspect,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
STOCK_SHA='79ac70eed826672f09a21df7e71c8646654e761961c9783ecd8e145c0a7149a7'
BINDING_SHA='8cc0e0abd677a087023e6cc2295d09a19c8d2086722fa5d9415cfe69e137d1e2'
UTILS_SHA='489c88ee233ee22cbcb75c8a36e68b67ce124c284254f9bc367a9b616898e70e'
SDK=Path('/opt/sglang/lib/python3.12/site-packages/nixl_cu13')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify_stock(stock):
 if stock.__name__!='nixl_cu13._api' or Path(stock.__file__).resolve()!=SDK/'_api.py' or sha(stock.__file__)!=STOCK_SHA:
  raise RuntimeError('Actual imported SDK wrapper differs')
 binding=stock.nixlBind
 if Path(binding.__file__).resolve()!=SDK/'_bindings.cpython-312-aarch64-linux-gnu.so' or sha(binding.__file__)!=BINDING_SHA or sha(SDK/'_utils.cpython-312-aarch64-linux-gnu.so')!=UTILS_SHA:
  raise RuntimeError('Actual native bindings/utils differ')
def load_overlay(stock):
 verify_stock(stock)
 path=HERE/'overlay/_api.py'
 # The runner independently pins every bundle byte before any agent; retain an
 # explicit self-contained wrapper pin for direct actor construction as well.
 if sha(path)!=OVERLAY_SHA:raise RuntimeError('Overlay bytes differ')
 name='nixl_cu13._api_poll_delay_r1';spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 for symbol in ('nixl_thread_sync_t','nixl_prepped_dlist_handle','nixl_xfer_handle','nixl_backend_handle'):
  setattr(module,symbol,getattr(stock,symbol))
 # Effective classes remain distinct and explicit. isinstance(stock) remains
 # true; unchanged inherited methods execute against the installed handle types.
 for symbol in ('nixl_agent_config','nixl_agent'):
  original=getattr(module,symbol)
  setattr(module,symbol,type(symbol,(getattr(stock,symbol),),{'__init__':original.__init__,'__module__':name}))
 if module.nixlBind is not stock.nixlBind:raise RuntimeError('Native module identity changed')
 sys.modules[name]=module
 return module

def actor_factories(progress,delay):
 if type(delay) is not int or delay not in (0,1000):raise ValueError('Only background true / delay0 or1000us')
 if progress.nixl_agent.__module__!='nixl_cu13._api':raise RuntimeError('Actor bridge requires untouched stock aliases; do not install twice')
 stock=sys.modules[progress.nixl_agent.__module__];overlay=load_overlay(stock)
 def config(**kw):
  if kw.get('num_threads')!=0 or kw.get('enable_listen_thread') is not True or kw.get('enable_prog_thread',True) is not True:
   raise ValueError('Keep background true, zero count and listener enabled')
  return overlay.nixl_agent_config(**dict(kw,enable_prog_thread=True,prog_thread_delay_us=delay))
 return config,overlay.nixl_agent,dict(stock_wrapper=str(stock.__file__),stock_wrapper_sha256=sha(stock.__file__),overlay_module=overlay.__name__,overlay_sha256=sha(HERE/'overlay/_api.py'),native_binding_sha256=BINDING_SHA,requested_delay_us=delay)

OVERLAY_SHA='472e81be7133e73d86efa2e3e71fa5a9dda46377d7886e9ecae8346713e2f150'
