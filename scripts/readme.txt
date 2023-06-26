Open terminal to use examples:



Pupil example 
    To run a standard eks:
    %run -i (or  %python) "scripts/pupil_example.py" --csv-dir 'data/ibl-pupil' --save-dir 'data/misc/pupil-test/' --diameter-s 0.99 --com-s 0.99 --eks_version "standard"
    To run an optimisation based eks:
    %run -i "scripts/pupil_example.py" --csv-dir 'data/ibl-pupil' --save-dir 'data/misc/pupil-test/' --diameter-s 0.99 --com-s 0.99 --eks_version "opti"
    
 

Mirror-mouse example
    %python scripts/multicam_example.py --csv-dir ./data/mirror-mouse --save-dir ./data/mirror-mouse/output --bodypart-list paw1LH paw2LF paw3RF paw4RH --camera-names top bot --eks_version opti
    %python scripts/multicam_example.py --csv-dir ./data/mirror-mouse --save-dir ./data/mirror-mouse/output --bodypart-list paw1LH paw2LF paw3RF paw4RH --camera-names top bot --eks_version other


Multiview paw example
    %python scripts/multiview_paw_example.py --csv-dir ./data/ibl-paw --save-dir ./outputs --eks_version opti
    %python scripts/multiview_paw_example.py --csv-dir ./data/ibl-paw --save-dir ./outputs --eks_version standard