import concurrent.futures, pathlib, subprocess, time
root=pathlib.Path('/data/projects/CloudRobo/habitat-gs')
text=subprocess.check_output(['git','config','--file','.gitmodules','--get-regexp','submodule.*.url'],cwd=root,text=True)
urls={l.split()[0].removeprefix('submodule.').removesuffix('.url'):l.split()[1] for l in text.splitlines()}
links=subprocess.check_output(['git','ls-tree','HEAD','src/deps/'],cwd=root,text=True)
jobs=[]
for line in links.splitlines():
 mode,kind,sha,path=line.split()
 if mode=='160000' and path not in ['src/deps/bullet3','src/deps/rlr-audio-propagation']:jobs.append((path,sha,urls[path]))
def run(job):
 path,sha,url=job;p=root/path;p.mkdir(exist_ok=True)
 with open(root.parent/'deployment/logs'/('fetch-'+p.name+'.log'),'w') as log:
  def cmd(*a):return subprocess.run(a,cwd=p,stdout=log,stderr=log).returncode
  if not (p/'.git').exists():cmd('git','init');cmd('git','remote','add','origin',url)
  for trial in range(3):
   if cmd('git','cat-file','-e',sha) and cmd('git','-c','http.version=HTTP/1.1','fetch','--depth','1','origin',sha):continue
   if not cmd('git','checkout','--detach',sha):
    print(path,'OK',flush=True);return
  raise RuntimeError(path)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 for result in ex.map(run,jobs):pass
