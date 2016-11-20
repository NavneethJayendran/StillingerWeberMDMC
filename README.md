# StillingerWeberMDMC
Project on usage of Markov Chain Monte Carlo Simulations to Study Phase Transitions of Stillinger-Weber Silicon Potential

### Getting Started

#### 1) Install Git
On Debian Linux (e.g. Ubuntu / Linux Mint), this can be done via:

`sudo apt-get update` <br /> 
`sudo apt-get install git`

For information on Mac or Windows installation, see [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

#### 2) Cloning the Repository
From a command line, cd into your directory of choice and run:

`git clone https://github.com/NavneethJayendran/StillingerWeberMDMC.git`

This will create a local copy of the entire Git repository on your machine. Any modifications you make will not be saved on this remote (online) repository. In order to save your changes, you must add and commit modified files. Git requires the extra add step so that you are less likely to commit dozens of unchecked changes. It's good practice to only add and commit files that are known to work properly (or, at the very least, don't fail outright).

#### 3) Making Changes

Once you modify files in your repository or create new ones, you will be given the option to commit them. Commits are essentially stored versions of your repository that you may use as backups in case you screw something up. But before committing, you must first stage changes using git add. First run:

`git status`

This will tell you what files you've modified since the last commit, and what files are not being tracked by Git. Files that are not being tracked and modified files will not be committed until you add them. For example, run:

`echo "This is a new file." > newfile.txt`  <br />
`git status`

Git will tell you that newfile.txt is not being tracked right now. In order to track it, run:

`git add newfile.txt`  <br />
`git status`

Git tells you that newfile.txt has been added but has not been committed. We'll get to that in the next step. For now, try modifying newfile.txt and check the status again:

`echo "This is a modified file." > newfile.txt`  <br />
`git status`

Git tells you that newfile.txt is already staged and ready to commit, but it also says newfile.txt has been modified and hasn't been staged for a commit. If you try to commit now, you will be committing the old version that read "This is a new file." instead of the new version which reads "This is a new file." This can be resolved by adding again.

`git add newfile.txt`  <br />
`git status`

And now you're ready to commit your changes.

#### 4) Committing Changes

Once your changes are staged using add, commit them via:

`git commit -m "A descriptive message"`

Git will probably complain that it doesn't know who you are. In order to mark the changes as your own, you need to set your global username or email for git using:

`git config --global user.email "you@example.com"`  <br />
`git config --global user.name "Your Name"`

Now try again.

`git commit -m "A descriptive message"` 

It should work at this point. However, note that committing changes does NOT save these changes to the remote repository. To do that, you must push.

#### 5) Pushing and Pulling Changes

These operations are a little bit intricate in the general case, so I'd advise reading a more comprehensive tutorial. But once you've made one or more commits, you can push (upload) those changes to the remote repository via:

`git push origin`

Now these changes should be updated online on the branch you've committed them to (read below for more info).

If you want to pull the latest changes from a given branch (read below for more info), use:

`git pull origin <branch_name>`

In some cases, you might run into a conflict if someone has modified a file and pushed those changes to the repository, and you try to push your own changes without taking theirs into account first. But we won't handle that for now, as merging is quite an arduous task (and this is more reason to use branches).

#### 6) (Optional ?) Creating a Branch

This may or may not be necessary. However, if multiple people want to work independently on the same file but want to keep it on the remote repository without having to constantly deal with each other's changes, the easiest way to do so is through branching.

Branches are essentially different versions of the same repository that can be worked on by different people. In order to work on a specific branch, go to your repository directory (e.g. on your own machine through the terminal) and run:

`git checkout <branch_name>`

The default and highest branch is called master. It's good practice to never modify master directly until a project has been finalized or at least reached a certain milestone in development (e.g. when a patch gets released for an open source tool). master is expected to be fully functional and stable by whoever chooses to download it. But to view it, on your own machine, simply do:

`git checkout master`

To create your own branch, go to your repository directory and run:

`git checkout -b <your_branch_name>`

For example, I created a branch called nav-dev.

`git checkout -b nav-dev`

In order to save the branch to the remote repository, run:

`git commit -m "Created new branch!"`

In order to check what branch you are currently working on, run:

`git branch`

At this point, it should show master, nav, nav-dev, and any other branches that were created since then and which you cloned. Feel free to make any changes you want on your own branch. No one will complain, since it leaves their branches unchanged.
