Last login: Tue Feb 26 23:17:31 on ttys000
dhcp-18-40-123-128:~ nehaprasad$ ssh nehap@18.114.1.93
nehap@18.114.1.93's password: 
Welcome to Ubuntu 16.04.4 LTS (GNU/Linux 4.15.0-42-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

251 packages can be updated.
0 updates are security updates.

New release '18.04.2 LTS' available.
Run 'do-release-upgrade' to upgrade to it.

*** System restart required ***
Last login: Wed Feb 27 18:16:19 2019 from 18.40.123.128
nehap@uhlergroup-desktop:~$ cd unbalanced_ot/code/geneexpression
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ ls
cell_metadata_in_vitro.txt     coordinates_in_vitro.txt    dat1  GAN.py                   main.py        __pycache__  run.sh          utils.py
clone_annotation_in_vitro.npz  counts_matrix_in_vitro.npz  dat2  gene_names_in_vitro.txt  preprocess.py  results      Untitled.ipynb
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote add origin git@github.com:uhlerlab/geneexpression.git
fatal: remote origin already exists.
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote -v
origin  https://pianolover@bitbucket.org/karreny/unbalanced_ot.git (fetch)
origin  https://pianolover@bitbucket.org/karreny/unbalanced_ot.git (push)
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote set-url git remote add origin git@github.com:uhlerlab/geneexpression.git
usage: git remote set-url [--push] <name> <newurl> [<oldurl>]
   or: git remote set-url --add <name> <newurl>
   or: git remote set-url --delete <name> <url>

    --push                manipulate push URLs
    --add                 add URL
    --delete              delete URLs

nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote set-url git@github.com:uhlerlab/geneexpression.git
usage: git remote set-url [--push] <name> <newurl> [<oldurl>]
   or: git remote set-url --add <name> <newurl>
   or: git remote set-url --delete <name> <url>

    --push                manipulate push URLs
    --add                 add URL
    --delete              delete URLs

nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote set-url --add origin git@github.com:uhlerlab/geneexpression.git
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote-v
git: 'remote-v' is not a git command. See 'git --help'.

Did you mean this?
    remote-fd
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote -v
origin  https://pianolover@bitbucket.org/karreny/unbalanced_ot.git (fetch)
origin  https://pianolover@bitbucket.org/karreny/unbalanced_ot.git (push)
origin  git@github.com:uhlerlab/geneexpression.git (push)
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git push -u origin git@github.com:uhlerlab/geneexpression.git
error: src refspec git@github.com does not match any.
error: failed to push some refs to 'https://pianolover@bitbucket.org/karreny/unbalanced_ot.git'
error: src refspec git@github.com does not match any.
error: failed to push some refs to 'git@github.com:uhlerlab/geneexpression.git'
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote set-url --add github git@github.com:uhlerlab/geneexpression.git
fatal: No such remote 'github'
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git push -u origin master
Password for 'https://pianolover@bitbucket.org': 
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote set-url --delete https://pianolover@bitbucket.org/karreny/unbalanced_ot.git
usage: git remote set-url [--push] <name> <newurl> [<oldurl>]
   or: git remote set-url --add <name> <newurl>
   or: git remote set-url --delete <name> <url>

    --push                manipulate push URLs
    --add                 add URL
    --delete              delete URLs

nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote rm ttps://pianolover@bitbucket.org/karreny/unbalanced_ot.git
error: Could not remove config section 'remote.ttps://pianolover@bitbucket.org/karreny/unbalanced_ot.git'
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ 
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote rm https://pianolover@bitbucket.org/karreny/unbalanced_ot.git
error: Could not remove config section 'remote.https://pianolover@bitbucket.org/karreny/unbalanced_ot.git'
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote rm origin
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote -v
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote set-url --add origin git@github.com:uhlerlab/geneexpression.git
fatal: No such remote 'origin'
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote set-url git@github.com:uhlerlab/geneexpression.git
usage: git remote set-url [--push] <name> <newurl> [<oldurl>]
   or: git remote set-url --add <name> <newurl>
   or: git remote set-url --delete <name> <url>

    --push                manipulate push URLs
    --add                 add URL
    --delete              delete URLs

nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote add origin git@github.com:uhlerlab/geneexpression.git
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote -v
origin  git@github.com:uhlerlab/geneexpression.git (fetch)
origin  git@github.com:uhlerlab/geneexpression.git (push)
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git push -u origin master
The authenticity of host 'github.com (192.30.253.113)' can't be established.
RSA key fingerprint is SHA256:nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8.
Are you sure you want to continue connecting (yes/no)? yes
Warning: Permanently added 'github.com,192.30.253.113' (RSA) to the list of known hosts.
Permission denied (publickey).
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ vim /root/.ssh/id_rsa.pub
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ cd ~
nehap@uhlergroup-desktop:~$ vim /root/.ssh/id_rsa.pub
nehap@uhlergroup-desktop:~$ ls
anaconda3  examples.desktop  unbalanced_ot
nehap@uhlergroup-desktop:~$ cd .ssh
nehap@uhlergroup-desktop:~/.ssh$ vim id_rsa.pub
nehap@uhlergroup-desktop:~/.ssh$ ls
known_hosts
nehap@uhlergroup-desktop:~/.ssh$ cd known_hosts
-bash: cd: known_hosts: Not a directory
nehap@uhlergroup-desktop:~/.ssh$ vim known_hosts
nehap@uhlergroup-desktop:~/.ssh$ ssh-keygen -t rsa -C nehap@mit.edu
Generating public/private rsa key pair.
Enter file in which to save the key (/home/nehap/.ssh/id_rsa): 
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in /home/nehap/.ssh/id_rsa.
Your public key has been saved in /home/nehap/.ssh/id_rsa.pub.
The key fingerprint is:
SHA256:aUmbyO+OO5x1ninJD9bfV8JwpipbrPB5doUaE3HtYdA nehap@mit.edu
The key's randomart image is:
+---[RSA 2048]----+
|            .+   |
|          . . E  |
|        .  o o . |
|     . o =. . +  |
|      o S  . B   |
|       o.o+ o + .|
|     ..++++B . ..|
|      +*==O...  .|
|      o+BB... .. |
+----[SHA256]-----+
nehap@uhlergroup-desktop:~/.ssh$ ls
id_rsa  id_rsa.pub  known_hosts
nehap@uhlergroup-desktop:~/.ssh$ vim id_rsa.pub
nehap@uhlergroup-desktop:~/.ssh$ cd ..
nehap@uhlergroup-desktop:~$ cd unbalanced_ot/code/geneexpression
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ ls
cell_metadata_in_vitro.txt     coordinates_in_vitro.txt    dat1  GAN.py                   main.py        __pycache__  run.sh          utils.py
clone_annotation_in_vitro.npz  counts_matrix_in_vitro.npz  dat2  gene_names_in_vitro.txt  preprocess.py  results      Untitled.ipynb
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git push -u origin master
Counting objects: 179, done.
Delta compression using up to 20 threads.
Compressing objects: 100% (94/94), done.
Writing objects: 100% (179/179), 8.37 MiB | 14.70 MiB/s, done.
Total 179 (delta 61), reused 179 (delta 61)
remote: Resolving deltas: 100% (61/61), done.
To git@github.com:uhlerlab/geneexpression.git
 * [new branch]      master -> master
Branch master set up to track remote branch master from origin.
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ ls
cell_metadata_in_vitro.txt     coordinates_in_vitro.txt    dat1  GAN.py                   main.py        __pycache__  run.sh          utils.py
clone_annotation_in_vitro.npz  counts_matrix_in_vitro.npz  dat2  gene_names_in_vitro.txt  preprocess.py  results      Untitled.ipynb
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git push origin master
Warning: Permanently added the RSA host key for IP address '192.30.253.112' to the list of known hosts.
Everything up-to-date
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add .
^C
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add
Nothing specified, nothing added.
Maybe you wanted to say 'git add .'?
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add .
^C
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ ls
cell_metadata_in_vitro.txt     coordinates_in_vitro.txt    dat1  GAN.py                   main.py        __pycache__  run.sh          utils.py
clone_annotation_in_vitro.npz  counts_matrix_in_vitro.npz  dat2  gene_names_in_vitro.txt  preprocess.py  results      Untitled.ipynb
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add cell_metadata_in_vitro.txt
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add clone_annotation_in_vitro.npz
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add coordinates_in_vitro.txt
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add counts_matrix_in_vitro.npz
^C
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add GAN.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add dat1
^C
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add main.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add preprocess.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add run.sh
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add utils.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add results/
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add dat2
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add dat1
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git commit -m "geneexpression files"

*** Please tell me who you are.

Run

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

to set your account's default identity.
Omit --global to set the identity only in this repository.

fatal: unable to auto-detect email address (got 'nehap@uhlergroup-desktop.(none)')
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git config --global user.email nehap@mit.edu
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git config --global user.name "Neha Prasad"
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git commit -m "geneexpression files"
[master 81e6bce] geneexpression files
 15 files changed, 393115 insertions(+)
 create mode 100644 code/geneexpression/GAN.py
 create mode 100644 code/geneexpression/cell_metadata_in_vitro.txt
 create mode 100644 code/geneexpression/clone_annotation_in_vitro.npz
 create mode 100644 code/geneexpression/coordinates_in_vitro.txt
 create mode 100644 code/geneexpression/dat1
 create mode 100644 code/geneexpression/dat2
 create mode 100644 code/geneexpression/main.py
 create mode 100644 code/geneexpression/preprocess.py
 create mode 100644 code/geneexpression/results/.DS_Store
 create mode 100644 code/geneexpression/results/exp4/geneexpression_exp4_netD.pth
 create mode 100644 code/geneexpression/results/exp4/geneexpression_exp4_netG.pth
 create mode 100644 code/geneexpression/results/exp4/geneexpression_exp4_tracker.pkl
 create mode 100644 code/geneexpression/results/exp4/log.txt
 create mode 100755 code/geneexpression/run.sh
 create mode 100644 code/geneexpression/utils.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git push origin master
Counting objects: 20, done.
Delta compression using up to 20 threads.
Compressing objects: 100% (19/19), done.
Writing objects: 100% (20/20), 182.64 MiB | 9.27 MiB/s, done.
Total 20 (delta 2), reused 0 (delta 0)
remote: Resolving deltas: 100% (2/2), completed with 1 local object.
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
remote: error: Trace: 8c15ae6fc8dea41c6dbc9488ef013a38
remote: error: See http://git.io/iEPt8g for more information.
remote: error: File code/geneexpression/dat1 is 116.83 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File code/geneexpression/dat2 is 116.33 MB; this exceeds GitHub's file size limit of 100.00 MB
To git@github.com:uhlerlab/geneexpression.git
 ! [remote rejected] master -> master (pre-receive hook declined)
error: failed to push some refs to 'git@github.com:uhlerlab/geneexpression.git'
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git reset dat1
Unstaged changes after reset:
M   code/toy_dataset/ubOT/run.sh
M   code/toy_dataset/ubOT/utils.py
M   code/zebrafish/ubOT/utils.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git reset dat2
Unstaged changes after reset:
M   code/toy_dataset/ubOT/run.sh
M   code/toy_dataset/ubOT/utils.py
M   code/zebrafish/ubOT/utils.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git push origin master
Counting objects: 20, done.
Delta compression using up to 20 threads.
Compressing objects: 100% (19/19), done.
Writing objects: 100% (20/20), 182.64 MiB | 8.34 MiB/s, done.
Total 20 (delta 2), reused 0 (delta 0)
remote: Resolving deltas: 100% (2/2), completed with 1 local object.
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
remote: error: Trace: 1fb6be49e4f17f42bb67d14a6a900e58
remote: error: See http://git.io/iEPt8g for more information.
remote: error: File code/geneexpression/dat1 is 116.83 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File code/geneexpression/dat2 is 116.33 MB; this exceeds GitHub's file size limit of 100.00 MB
To git@github.com:uhlerlab/geneexpression.git
 ! [remote rejected] master -> master (pre-receive hook declined)
error: failed to push some refs to 'git@github.com:uhlerlab/geneexpression.git'
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git reset HEAD dat1
Unstaged changes after reset:
M   code/toy_dataset/ubOT/run.sh
M   code/toy_dataset/ubOT/utils.py
M   code/zebrafish/ubOT/utils.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git rm dat1
rm 'code/geneexpression/dat1'
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git rm dat2
rm 'code/geneexpression/dat2'
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git status
On branch master
Your branch is ahead of 'origin/master' by 1 commit.
  (use "git push" to publish your local commits)
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

    deleted:    dat1
    deleted:    dat2

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

    modified:   ../toy_dataset/ubOT/run.sh
    modified:   ../toy_dataset/ubOT/utils.py
    modified:   ../zebrafish/ubOT/utils.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)

    ../../.DS_Store
    ../.DS_Store
    ../geneexpression.zip
    .DS_Store
    .ipynb_checkpoints/
    Untitled.ipynb
    __pycache__/
    counts_matrix_in_vitro.npz
    gene_names_in_vitro.txt
    ../toy_dataset/ubOT/.DS_Store
    ../toy_dataset/ubOT/__pycache__/
    ../toy_dataset/ubOT/results/
    ../zebrafish/ubOT/.DS_Store
    ../zebrafish/ubOT/__pycache__/
    ../zebrafish/ubOT/results/
    ../../datasets/toy_dataset/.DS_Store
    ../../datasets/toy_dataset/data/

nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git commit -m "delete dat1 and dat2 since they are too big"
[master 9d1e73f] delete dat1 and dat2 since they are too big
 2 files changed, 0 insertions(+), 0 deletions(-)
 delete mode 100644 code/geneexpression/dat1
 delete mode 100644 code/geneexpression/dat2
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ ls
cell_metadata_in_vitro.txt     coordinates_in_vitro.txt    GAN.py                   main.py        __pycache__  run.sh          utils.py
clone_annotation_in_vitro.npz  counts_matrix_in_vitro.npz  gene_names_in_vitro.txt  preprocess.py  results      Untitled.ipynb
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add cell_metadata_in_vitro.txt
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add coordinates_in_vitro.txt
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add clone_annotation_in_vitro.npz
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add GAN.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add gene_names_in_vitro.txt
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add main.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add results
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add preprocess.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add run.sh
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add utils.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git commit -m "add files"
[master 994b937] add files
 1 file changed, 25289 insertions(+)
 create mode 100644 code/geneexpression/gene_names_in_vitro.txt
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git push origin master
Counting objects: 29, done.
Delta compression using up to 20 threads.
Compressing objects: 100% (28/28), done.
Writing objects: 100% (29/29), 182.71 MiB | 8.48 MiB/s, done.
Total 29 (delta 6), reused 0 (delta 0)
remote: Resolving deltas: 100% (6/6), completed with 1 local object.
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
remote: error: Trace: 0585f49be1bab30482b8a5b0645e12d1
remote: error: See http://git.io/iEPt8g for more information.
remote: error: File code/geneexpression/dat1 is 116.83 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File code/geneexpression/dat2 is 116.33 MB; this exceeds GitHub's file size limit of 100.00 MB
To git@github.com:uhlerlab/geneexpression.git
 ! [remote rejected] master -> master (pre-receive hook declined)
error: failed to push some refs to 'git@github.com:uhlerlab/geneexpression.git'
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git reset --hard origin
fatal: ambiguous argument 'origin': unknown revision or path not in the working tree.
Use '--' to separate paths from revisions, like this:
'git <command> [<revision>...] -- [<file>...]'
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git reset --hard origin/master
HEAD is now at 84f43fd small edit
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git status
On branch master
Your branch is up-to-date with 'origin/master'.
Untracked files:
  (use "git add <file>..." to include in what will be committed)

    ../../.DS_Store
    ../.DS_Store
    ../geneexpression.zip
    ./
    ../toy_dataset/ubOT/.DS_Store
    ../toy_dataset/ubOT/__pycache__/
    ../toy_dataset/ubOT/results/
    ../zebrafish/ubOT/.DS_Store
    ../zebrafish/ubOT/__pycache__/
    ../zebrafish/ubOT/results/
    ../../datasets/toy_dataset/.DS_Store
    ../../datasets/toy_dataset/data/

nothing added to commit but untracked files present (use "git add" to track)
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ ls
counts_matrix_in_vitro.npz  __pycache__  Untitled.ipynb
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ ls
counts_matrix_in_vitro.npz  __pycache__  Untitled.ipynb
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git commit
On branch master
Your branch is up-to-date with 'origin/master'.
Untracked files:
    ../../.DS_Store
    ../.DS_Store
    ../geneexpression.zip
    ./
    ../toy_dataset/ubOT/.DS_Store
    ../toy_dataset/ubOT/__pycache__/
    ../toy_dataset/ubOT/results/
    ../zebrafish/ubOT/.DS_Store
    ../zebrafish/ubOT/__pycache__/
    ../zebrafish/ubOT/results/
    ../../datasets/toy_dataset/.DS_Store
    ../../datasets/toy_dataset/data/

nothing added to commit but untracked files present
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ cd ..
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ ls
geneexpression      mnist          toy_dataset        ub_mnist_bright2  ub_mnist_perm_adj         ub_mnist_usps_perm    zebrafish
geneexpression.zip  mnist_usps_AE  ub_celebA_age_256  ub_mnist_perm     ub_mnist_perm_bri_smooth  ub_NUM_perm_smooth_3
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ cd geneexpression
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ ls
counts_matrix_in_vitro.npz  __pycache__  Untitled.ipynb
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git reflog
84f43fd HEAD@{0}: reset: moving to origin/master
994b937 HEAD@{1}: commit: add files
9d1e73f HEAD@{2}: commit: delete dat1 and dat2 since they are too big
81e6bce HEAD@{3}: commit: geneexpression files
84f43fd HEAD@{4}: clone: from https://pianolover@bitbucket.org/karreny/unbalanced_ot.git
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git reset --hard 81e6bce
Checking out files: 100% (15/15), done.
HEAD is now at 81e6bce geneexpression files
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ ls
cell_metadata_in_vitro.txt     coordinates_in_vitro.txt    dat1  GAN.py   preprocess.py  results  Untitled.ipynb
clone_annotation_in_vitro.npz  counts_matrix_in_vitro.npz  dat2  main.py  __pycache__    run.sh   utils.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add GAN.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add main.py
ginehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add preprocess.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add utils.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add run.sh
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add cell_metadata_in_vitro.txt
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git commit -m "add gene expression files"
On branch master
Your branch is ahead of 'origin/master' by 1 commit.
  (use "git push" to publish your local commits)
Untracked files:
    ../../.DS_Store
    ../.DS_Store
    ../geneexpression.zip
    .DS_Store
    .ipynb_checkpoints/
    Untitled.ipynb
    __pycache__/
    counts_matrix_in_vitro.npz
    ../toy_dataset/ubOT/.DS_Store
    ../toy_dataset/ubOT/__pycache__/
    ../toy_dataset/ubOT/results/
    ../zebrafish/ubOT/.DS_Store
    ../zebrafish/ubOT/__pycache__/
    ../zebrafish/ubOT/results/
    ../../datasets/toy_dataset/.DS_Store
    ../../datasets/toy_dataset/data/

nothing added to commit but untracked files present
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git push origin master
Counting objects: 20, done.
Delta compression using up to 20 threads.
Compressing objects: 100% (19/19), done.
Writing objects: 100% (20/20), 182.64 MiB | 11.02 MiB/s, done.
Total 20 (delta 2), reused 0 (delta 0)
remote: Resolving deltas: 100% (2/2), completed with 1 local object.
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
remote: error: Trace: c40f4ae024639a4fede735a82155ecc8
remote: error: See http://git.io/iEPt8g for more information.
remote: error: File code/geneexpression/dat1 is 116.83 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File code/geneexpression/dat2 is 116.33 MB; this exceeds GitHub's file size limit of 100.00 MB
To git@github.com:uhlerlab/geneexpression.git
 ! [remote rejected] master -> master (pre-receive hook declined)
error: failed to push some refs to 'git@github.com:uhlerlab/geneexpression.git'
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ ls
cell_metadata_in_vitro.txt     coordinates_in_vitro.txt    dat1  GAN.py   preprocess.py  results  Untitled.ipynb
clone_annotation_in_vitro.npz  counts_matrix_in_vitro.npz  dat2  main.py  __pycache__    run.sh   utils.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git rm dat1
rm 'code/geneexpression/dat1'
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git rm dat2
rm 'code/geneexpression/dat2'
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ ls
cell_metadata_in_vitro.txt     coordinates_in_vitro.txt    GAN.py   preprocess.py  results  Untitled.ipynb
clone_annotation_in_vitro.npz  counts_matrix_in_vitro.npz  main.py  __pycache__    run.sh   utils.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add dat1
fatal: pathspec 'dat1' did not match any files
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git push origin master
Counting objects: 20, done.
Delta compression using up to 20 threads.
Compressing objects: 100% (19/19), done.
Writing objects: 100% (20/20), 182.64 MiB | 9.80 MiB/s, done.
Total 20 (delta 2), reused 0 (delta 0)
remote: Resolving deltas: 100% (2/2), completed with 1 local object.
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
remote: error: Trace: 4c486fe4f6d130ea0386006590c48df7
remote: error: See http://git.io/iEPt8g for more information.
remote: error: File code/geneexpression/dat1 is 116.83 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File code/geneexpression/dat2 is 116.33 MB; this exceeds GitHub's file size limit of 100.00 MB
To git@github.com:uhlerlab/geneexpression.git
 ! [remote rejected] master -> master (pre-receive hook declined)
error: failed to push some refs to 'git@github.com:uhlerlab/geneexpression.git'
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ cd ..
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ ls
geneexpression      mnist          toy_dataset        ub_mnist_bright2  ub_mnist_perm_adj         ub_mnist_usps_perm    zebrafish
geneexpression.zip  mnist_usps_AE  ub_celebA_age_256  ub_mnist_perm     ub_mnist_perm_bri_smooth  ub_NUM_perm_smooth_3
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ git add geneexpression/GAN.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ git commit -m "add gan.py"
[master f3e849f] add gan.py
 2 files changed, 0 insertions(+), 0 deletions(-)
 delete mode 100644 code/geneexpression/dat1
 delete mode 100644 code/geneexpression/dat2
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ ls
geneexpression      mnist          toy_dataset        ub_mnist_bright2  ub_mnist_perm_adj         ub_mnist_usps_perm    zebrafish
geneexpression.zip  mnist_usps_AE  ub_celebA_age_256  ub_mnist_perm     ub_mnist_perm_bri_smooth  ub_NUM_perm_smooth_3
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ cd geneexpression
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ ls
cell_metadata_in_vitro.txt     coordinates_in_vitro.txt    GAN.py   preprocess.py  results  Untitled.ipynb
clone_annotation_in_vitro.npz  counts_matrix_in_vitro.npz  main.py  __pycache__    run.sh   utils.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ cd ..
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ git reflog
f3e849f HEAD@{0}: commit: add gan.py
81e6bce HEAD@{1}: reset: moving to 81e6bce
84f43fd HEAD@{2}: reset: moving to origin/master
994b937 HEAD@{3}: commit: add files
9d1e73f HEAD@{4}: commit: delete dat1 and dat2 since they are too big
81e6bce HEAD@{5}: commit: geneexpression files
84f43fd HEAD@{6}: clone: from https://pianolover@bitbucket.org/karreny/unbalanced_ot.git
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ git push origin f3e849f:master
Counting objects: 24, done.
Delta compression using up to 20 threads.
Compressing objects: 100% (23/23), done.
Writing objects: 100% (24/24), 182.64 MiB | 9.76 MiB/s, done.
Total 24 (delta 4), reused 0 (delta 0)
remote: Resolving deltas: 100% (4/4), completed with 1 local object.
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
remote: error: Trace: 6bc4d2ebff7ff56af0e89bcba52f9adb
remote: error: See http://git.io/iEPt8g for more information.
remote: error: File code/geneexpression/dat1 is 116.83 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File code/geneexpression/dat2 is 116.33 MB; this exceeds GitHub's file size limit of 100.00 MB
To git@github.com:uhlerlab/geneexpression.git
 ! [remote rejected] f3e849f -> master (pre-receive hook declined)
error: failed to push some refs to 'git@github.com:uhlerlab/geneexpression.git'
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ git ls-remote <url> <refs>
-bash: syntax error near unexpected token `<'
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ git log | head -n 1 | awk '{print $2}'
f3e849f8999391cf01938e1ca4034b7514ee1418
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ git push origin f3e849f8999391cf01938e1ca4034b7514ee1418:master
Counting objects: 24, done.
Delta compression using up to 20 threads.
Compressing objects: 100% (23/23), done.
Writing objects: 100% (24/24), 182.64 MiB | 10.51 MiB/s, done.
Total 24 (delta 4), reused 0 (delta 0)
remote: Resolving deltas: 100% (4/4), completed with 1 local object.
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
remote: error: Trace: fef41c3ea093c24cb77e88a167db9759
remote: error: See http://git.io/iEPt8g for more information.
remote: error: File code/geneexpression/dat1 is 116.83 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File code/geneexpression/dat2 is 116.33 MB; this exceeds GitHub's file size limit of 100.00 MB
To git@github.com:uhlerlab/geneexpression.git
 ! [remote rejected] f3e849f8999391cf01938e1ca4034b7514ee1418 -> master (pre-receive hook declined)
error: failed to push some refs to 'git@github.com:uhlerlab/geneexpression.git'
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ ls
geneexpression      mnist          toy_dataset        ub_mnist_bright2  ub_mnist_perm_adj         ub_mnist_usps_perm    zebrafish
geneexpression.zip  mnist_usps_AE  ub_celebA_age_256  ub_mnist_perm     ub_mnist_perm_bri_smooth  ub_NUM_perm_smooth_3
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ cd geneexpression
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ ls
cell_metadata_in_vitro.txt     coordinates_in_vitro.txt    GAN.py   preprocess.py  results  Untitled.ipynb
clone_annotation_in_vitro.npz  counts_matrix_in_vitro.npz  main.py  __pycache__    run.sh   utils.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ cd ..
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ cd geneexpression
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ ls
cell_metadata_in_vitro.txt     coordinates_in_vitro.txt    GAN.py   preprocess.py  results  Untitled.ipynb
clone_annotation_in_vitro.npz  counts_matrix_in_vitro.npz  main.py  __pycache__    run.sh   utils.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git remote -v
origin  git@github.com:uhlerlab/geneexpression.git (fetch)
origin  git@github.com:uhlerlab/geneexpression.git (push)
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ 
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git add main.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git status
On branch master
Your branch is ahead of 'origin/master' by 2 commits.
  (use "git push" to publish your local commits)
Untracked files:
  (use "git add <file>..." to include in what will be committed)

    ../../.DS_Store
    ../.DS_Store
    ../geneexpression.zip
    .DS_Store
    .ipynb_checkpoints/
    Untitled.ipynb
    __pycache__/
    counts_matrix_in_vitro.npz
    ../toy_dataset/ubOT/.DS_Store
    ../toy_dataset/ubOT/__pycache__/
    ../toy_dataset/ubOT/results/
    ../zebrafish/ubOT/.DS_Store
    ../zebrafish/ubOT/__pycache__/
    ../zebrafish/ubOT/results/
    ../../datasets/toy_dataset/.DS_Store
    ../../datasets/toy_dataset/data/

nothing added to commit but untracked files present (use "git add" to track)
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ grep dat1
^C
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ git reflog
f3e849f HEAD@{0}: commit: add gan.py
81e6bce HEAD@{1}: reset: moving to 81e6bce
84f43fd HEAD@{2}: reset: moving to origin/master
994b937 HEAD@{3}: commit: add files
9d1e73f HEAD@{4}: commit: delete dat1 and dat2 since they are too big
81e6bce HEAD@{5}: commit: geneexpression files
84f43fd HEAD@{6}: clone: from https://pianolover@bitbucket.org/karreny/unbalanced_ot.git
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ cd ..
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ ls
geneexpression      mnist          toy_dataset        ub_mnist_bright2  ub_mnist_perm_adj         ub_mnist_usps_perm    zebrafish
geneexpression.zip  mnist_usps_AE  ub_celebA_age_256  ub_mnist_perm     ub_mnist_perm_bri_smooth  ub_NUM_perm_smooth_3
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ cd geneexpression
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ l
cell_metadata_in_vitro.txt     coordinates_in_vitro.txt    GAN.py   preprocess.py  results/  Untitled.ipynb
clone_annotation_in_vitro.npz  counts_matrix_in_vitro.npz  main.py  __pycache__/   run.sh*   utils.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ ls
cell_metadata_in_vitro.txt     coordinates_in_vitro.txt    GAN.py   preprocess.py  results  Untitled.ipynb
clone_annotation_in_vitro.npz  counts_matrix_in_vitro.npz  main.py  __pycache__    run.sh   utils.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ cd ..
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ git add geneexpression/gan.py
fatal: pathspec 'geneexpression/gan.py' did not match any files
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ git add geneexpression/GAN.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ git status
On branch master
Your branch is ahead of 'origin/master' by 2 commits.
  (use "git push" to publish your local commits)
Untracked files:
  (use "git add <file>..." to include in what will be committed)

    ../.DS_Store
    .DS_Store
    geneexpression.zip
    geneexpression/.DS_Store
    geneexpression/.ipynb_checkpoints/
    geneexpression/Untitled.ipynb
    geneexpression/__pycache__/
    geneexpression/counts_matrix_in_vitro.npz
    toy_dataset/ubOT/.DS_Store
    toy_dataset/ubOT/__pycache__/
    toy_dataset/ubOT/results/
    zebrafish/ubOT/.DS_Store
    zebrafish/ubOT/__pycache__/
    zebrafish/ubOT/results/
    ../datasets/toy_dataset/.DS_Store
    ../datasets/toy_dataset/data/

nothing added to commit but untracked files present (use "git add" to track)
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ git commit -m "added gan.py"
On branch master
Your branch is ahead of 'origin/master' by 2 commits.
  (use "git push" to publish your local commits)
Untracked files:
    ../.DS_Store
    .DS_Store
    geneexpression.zip
    geneexpression/.DS_Store
    geneexpression/.ipynb_checkpoints/
    geneexpression/Untitled.ipynb
    geneexpression/__pycache__/
    geneexpression/counts_matrix_in_vitro.npz
    toy_dataset/ubOT/.DS_Store
    toy_dataset/ubOT/__pycache__/
    toy_dataset/ubOT/results/
    zebrafish/ubOT/.DS_Store
    zebrafish/ubOT/__pycache__/
    zebrafish/ubOT/results/
    ../datasets/toy_dataset/.DS_Store
    ../datasets/toy_dataset/data/

nothing added to commit but untracked files present
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ git status
On branch master
Your branch is ahead of 'origin/master' by 2 commits.
  (use "git push" to publish your local commits)
Untracked files:
  (use "git add <file>..." to include in what will be committed)

    ../.DS_Store
    .DS_Store
    geneexpression.zip
    geneexpression/.DS_Store
    geneexpression/.ipynb_checkpoints/
    geneexpression/Untitled.ipynb
    geneexpression/__pycache__/
    geneexpression/counts_matrix_in_vitro.npz
    toy_dataset/ubOT/.DS_Store
    toy_dataset/ubOT/__pycache__/
    toy_dataset/ubOT/results/
    zebrafish/ubOT/.DS_Store
    zebrafish/ubOT/__pycache__/
    zebrafish/ubOT/results/
    ../datasets/toy_dataset/.DS_Store
    ../datasets/toy_dataset/data/

nothing added to commit but untracked files present (use "git add" to track)
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ ls
geneexpression      mnist          toy_dataset        ub_mnist_bright2  ub_mnist_perm_adj         ub_mnist_usps_perm    zebrafish
geneexpression.zip  mnist_usps_AE  ub_celebA_age_256  ub_mnist_perm     ub_mnist_perm_bri_smooth  ub_NUM_perm_smooth_3
nehap@uhlergroup-desktop:~/unbalanced_ot/code$ cd geneexpression
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ ls
cell_metadata_in_vitro.txt     coordinates_in_vitro.txt    GAN.py   preprocess.py  results  Untitled.ipynb
clone_annotation_in_vitro.npz  counts_matrix_in_vitro.npz  main.py  __pycache__    run.sh   utils.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ vim GAN.py
nehap@uhlergroup-desktop:~/unbalanced_ot/code/geneexpression$ vim main.py

import torch
from torch import nn, optim
from torch.autograd import Variable, grad

import GAN
import utils
import visdom

import numpy as np
import sys
import os
import pickle

torch.manual_seed(1)

#============ PARSE ARGUMENTS =============

args = utils.setup_args()
args.save_name = args.save_file + args.env
print(args)

#============ GRADIENT PENALTY (for discriminator) ================

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size())

    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if torch.cuda.is_available():
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.lambG
    return gradient_penalty

def calc_gradient_penalty_rho(netD, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size())

    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if torch.cuda.is_available():
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    _, disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.lambG2
    return gradient_penalty



#============= TRAINING INITIALIZATION ==============

# initialize discriminator
netD = GAN.Discriminator(args.nz, args.n_hidden)
print("Discriminator loaded")

# initialize generator
netG = GAN.Generator(args.nz, args.n_hidden)
print("Generator loaded")

if torch.cuda.is_available():
    netD.cuda()
    netG.cuda()
    print("Using GPU")

# load data
loader = utils.setup_data_loaders(args.batch_size)
print('Data loaded')
sys.stdout.flush()

# setup optimizers
G_opt = optim.Adam(list(netG.parameters()), lr = args.lrG)
D_opt = optim.Adam(list(netD.parameters()), lr = args.lrD)

# loss criteria
logsigmoid = nn.LogSigmoid()
mse = nn.MSELoss(reduce=False)
LOG2 = Variable(torch.from_numpy(np.ones(1)*np.log(2)).float())
print(LOG2)
if torch.cuda.is_available():
    LOG2 = LOG2.cuda()

#=========== LOGGING INITIALIZATION ================

# vis = utils.init_visdom(args.env)
# tracker = utils.Tracker()
# tracker_plot=None
# scale_plot=None

#============================================================
#============ MAIN TRAINING LOOP ============================
#============================================================

for epoch in range(args.max_iter):
    for it, (s_inputs, t_inputs) in enumerate(loader):

        s_inputs, t_inputs = Variable(s_inputs), Variable(t_inputs)
        if torch.cuda.is_available():
            s_inputs, t_inputs = s_inputs.cuda(), t_inputs.cuda()

#================== Train generator =========================
        if it % args.critic_iter == args.critic_iter-1:
            netG.train()
            netD.eval()

            netG.zero_grad()

            # pass source inputs through generator network
            s_generated, s_scale = netG(s_inputs)

            # pass generated source data and target inputs through discriminator network
            s_outputs = netD(s_generated)

            # compute loss
#            G_loss = torch.mean(s_scale*torch.sum(mse(s_generated, s_inputs), dim=1)) + args.lamb1*torch.mean(-torch.log(s_scale)+s_scale)
#            G_loss = torch.mean(s_scale*torch.sum(mse(s_generated, s_inputs), dim=1)) - args.lamb1*torch.mean(s_scale*(1-torch.log(s_scale)))
            G_loss = args.lamb0*torch.mean(s_scale*torch.sum(mse(s_generated, s_inputs), dim=1)) + args.lamb1*torch.mean((s_scale-1)*torch.log(s_scale))
           G_loss += calc_gradient_penalty_rho(netG, s_inputs.data, s_inputs.data[torch.randperm(s_inputs.size(0))])

            if args.psi2 == "EQ":
                G_loss += - args.lamb2*torch.mean(s_scale*s_outputs)
            else:
                G_loss += args.lamb2*torch.mean(s_scale*(LOG2.expand_as(s_outputs)+logsigmoid(s_outputs)-s_outputs))

            # update params
            G_loss.backward()
            G_opt.step()


#================== Train discriminator =========================
        else:
            netD.train()
            netG.eval()

            netD.zero_grad()

            # pass source inputs through generator network
            s_generated, s_scale = netG(s_inputs)

            # pass generated source data and target inputs through discriminator network
            s_outputs, t_outputs = netD(s_generated), netD(t_inputs)

            # compute loss
            #D_loss = 0
            D_loss = calc_gradient_penalty(netD, s_generated.data, t_inputs.data)
            if args.psi2 == "EQ":
                D_loss += torch.mean(s_scale*s_outputs) - torch.mean(t_outputs)
            else:
                D_loss += -torch.mean(s_scale*(LOG2.expand_as(s_outputs)+logsigmoid(s_outputs)-s_outputs)) - torch.mean(LOG2.expand_as(t_outputs)+logsigmoid(t_outputs)) #+ calc_gradient_penalty(netD, s_generated.data, t_inputs.data)

            # update params
            D_loss.backward()
            D_opt.step()


#================= Log results ===========================================
    
    netD.eval()
    netG.eval()

    for s_inputs, t_inputs in loader:
        num = s_inputs.size(0)
        s_inputs, t_inputs = Variable(s_inputs), Variable(t_inputs)
        if torch.cuda.is_available():
            s_inputs, t_inputs = s_inputs.cuda(), t_inputs.cuda()

        s_generated, s_scale = netG(s_inputs)
        s_outputs, t_outputs = netD(s_generated), netD(t_inputs)

        # update tracker
        W_loss = args.lamb0*torch.mean(s_scale*torch.sum(mse(s_generated, s_inputs), dim=1)) + args.lamb1*torch.mean(-torch.log(s_scale)+s_scale)
        W_loss += torch.mean(s_scale*(LOG2.expand_as(s_outputs)+logsigmoid(s_outputs)-s_outputs))
        W_loss += torch.mean(LOG2.expand_as(t_outputs)+logsigmoid(t_outputs))
        #tracker.add(W_loss.cpu().data, num)

    #tracker.tick()

    # save models
    torch.save(netD.cpu().state_dict(), args.save_name+"_netD.pth")
    torch.save(netG.cpu().state_dict(), args.save_name+"_netG.pth")

    if torch.cuda.is_available():
        netD.cuda()
        netG.cuda()

    # save tracker
   # with open(args.save_name+"_tracker.pkl", 'wb') as f:
    #    pickle.dump(tracker, f)

    # if epoch % 100 == 0:
    #     tracker_plot, scale_plot = utils.plot(tracker, tracker_plot, scale_plot, s_scale.cpu().data.numpy(), args.env, vis)
