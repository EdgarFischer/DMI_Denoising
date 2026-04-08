nohup env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.25 \
/bilbo/usr/local/matlab2017b/bin/matlab \
-nodisplay -nosplash -nodesktop \
-r "run('tmppcaProton.m'); exit;" \
> matlab_tmppca.log 2>&1 &
