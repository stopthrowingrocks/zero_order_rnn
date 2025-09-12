# Trying to get the repo working
## Trying to provision an instance
Decision: reading the docs. I want to make sure that I'm doing the right thing and following the code correctly.

Maybe it would just be better to try and provision an instance and just eat like $3 to move a bit faster. If it runs into problems then maybe we can look at the docs. Decision: don't read the docs and just provision an instance.

### Decision point: Purchase the $3.20/hr on-demand 8xA40 instance group or the $1.60/hr interruptible instance group.
Suppose
- We purchase the $3.20/hr one
- it takes 4 hours to get all this working
- we would not have encountered any problems with the $1.60/hr option. Yes, I am also assuming that we are buying the $3.20/hr one because right now I think that's the best option and I want to explain the reasons why.

In total this would come out to like $6 wasted. Are you ok with that?

I honestly have no idea. The money is draining so quickly in other areas that this is probably not where I would fight back.

Where would you fight back?

Like the bike and the car, or the Stanford apartment, or another apartment. You're thinking about it way too much for me to feel safe right now.

Look, we have money. We have a lot of money. After paying for Stanford we will have $120K in our account, not to mention all the money that Mom has in the custodial account for us.

But if you start spending money without a solid plan in place the money could just disappear one day and you wake up and are really in the shits.

But this is not the thing that will kill us. After this purchase we will still have more than $100K.

Again, it's not any individual purchase that I have a problem with. It's the fact that purchases don't seem to affect you, while gains are looked at with great happiness and excitement.

Well yeah the ability to earn money is is nice thing. It's not a linear scale. When you have the ability to make a lot of money at the drop of a hat (provided that we had enough stuff figured out to actually hold down a solid job making a solid income) then expenses don't really matter. Maybe the excitement at earning money is because it is initial proof of this money-making ability that doesn't really need to be directly developed.

Block.

Just to be clear, the next decision point is deciding whether to get the $3.20/hr instance or the $1.60/hr instance, right? Is there even a world where we get the $1.60/hr one? You'd make our life harder to save $6?

Block.

I just think the $6 saving is a bad idea.

I know.

And you want to do it anyways?

That's not what I'm saying. You don't really have systems to track your spending. I know you've been working on your stock simulator but you've really been taking your time with it. I know you're enjoying learning about pandas and matplotlib and exploring the data and trying to optimize it and make sure it's all correct but it doesn't even work right now. If you wanted to just get it out and be safe you could just like double the volatilities and fudge the correlation coefficients to make it less in your favor. I'm also not saying you should do it now, it's not really my job to decide that.

Would you rather me just get the cheaper instance right now?

I recognize you're kind of in a bind right now. You want to get the repo working by tomorrow and you don't even know if you can. You have no idea how much work it takes and you're kinda hoping it's either not that bad or you can come to Mykel tomorrow with specific problems to resolve but either way you feel like you need to get working on it and you perceive me as just another obstacle to overcome. At the same time you can see how that puts me in a bad spot because although I'm trying to help you out it doesn't feel like that right now. And you go on thinking "Well I've got all this money, I've gotta use it for *something*." What if you schedule time tomorrow to allocate to budgeting? We don't need to think about exactly what that looks like right now, just time on the calendar. An hour is all I ask for now. Also, a hard limit of what you're willing to spend today. How does $200 sound?

I think if I'm spending more than $100 setting it up I should rethink my approach and I should ask Mykel about it. "Am I expected to put my own money to this degree into the project? Am I doing something wrong or inefficiently or is this just what it takes to do work of this kind?" How about $150?

How about $100?

How about $120? I feel like if I can't get it done with $120 then I should ask for help.

Let's pay the Stanford bill first because right now that's hanging over our heads. ... Alright part of it is paid but we will need to sell some stocks first. ... We can't sell the stocks right now. We will have to sell them at least tomorrow because this is to get money and I don't want to lose a lot on the trade. What about just selling 8 shares of SPY, you say? Under the logic that "it's just the market anyways." idk I just think that selling with a bit more intention is better. Yes, I am asking for time tomorrow to do stuff. Shouldn't it not be that hard to just get on the computer and sell the shares?

What if when we get on the computer we play videogames instead? Then nobody wins.

Ok, we can sell 8 shares now. Hell, let's just sell 9. And let's do a limit order and just go with Schwab's recommended price. [What if the price drops a whole bunch and we can't sell?] Then I think we have different problems. ... Bought. We will also need to move the money into the savings account once the stocks are sold and then we can pay off the rest of the Stanford bill.

So how do you feel about a $120 budget until we talk to Mykel Kochenderfer tomorrow? If we don't get to talk to him tomorrow we can revisit this and discuss again.

I probably don't want to spend $500 without talking to him but we can do the math again if we really need to.

Sure. So how do you feel about a $120 budget?

Remind me again what this is for?

This is to provision GPUs and other resources for the Stanford Capstone project. This is intended to result in graduating from Stanford, developing a working relationship with a Stanford professor or faculty member, and getting experience working on Machine Learning architectures. All of this is in service of career so that we can earn more money in the future, and I think it is one of our best realistic short-term bets on making money in a way that is also interesting and enjoyable and has the potential to earn a lot.

Ok, although I am scared, I will permit a $120 budget until we talk with Mykel tomorrow at 1:45pm. Even assuming that all the money gets used (which I know you would say is unlikely), it seems like a small amount and it is also somewhat directly in service of making more money in the future. So yes, you are permitted a $120 budget until 1:45pm tomorrow.

Ok, I'm going to take your word for it for now. Now $3.20/hr means that we would get about 30-40 hours of 8xA40 time which is more than plenty for now. **I will start with buying the 8xA40 on-demand instance and then just try to get things working**, and I won't care about wasting a bit of money if it gives me the peace of mind to work on other parts of the puzzle. Yes, it would be cool to save documents like this and feed them to an LLM later. Because sticking to the budget is important, I wonder if there is a way to limit the amount of time it's up for.

### Idea: Use Instant Clusters instead of provisioning the nodes myself
Ok look, let's just stick to what the docs say to use. It is way easier to get help from the professor than from the runpod people.

## Getting back to work
I need to get an instance with an SSH key which means that I need to switch pages to the settings page. Once I am done with the settings page, I can always continue making progress by returning to the "Deploy a Pod" page.

I am also getting hungry. Let's put the shrimp in the oven. ... The shrimp was half-eaten so I put the handheld pot pies in the oven instead. On the way back I glanced over at the piano. I would like to keep the routine of playing the piano for an hour everyday. I would also like to keep the routines of working out outside, going to yoga and rock climbing, not to mention the basic ones of brushing my teeth and showering everyday. Also my shirt is dirty and my room is messy and I'm hungry and maybe I should just eat something right now while the pot pies are cooking. Maybe I could play the piano now while the food cooks. Also in the oven was the pesto chicken that I'd left in the other day and in the interest of food safety I threw it out but it was an unfortunate waste and I wish that things like that didn't happen but they do and it just seems like there are too many things that I want to change in my life and I just don't have enough time/energy/focus/alignment/integration or whatever in order to do them or at least be effective at working through them. When it comes to material improvement it seems like the creative/learning side is such a drag.

Ok here's an argument for doing the piano. You won't be as effective at working while you're hungry.

Fair enough but we can provision the GPUs first. It is in the budget after all. And I appreciate that we made the budget because now, according to that agreement, money doesn't matter as long as it fits within the budget. That is, after all, what the budget means. And it gives the rest of us space to resolve our differences.

I understand and I can see what you're saying but I'm not going to budge on asking you to allocate time to a more thorough budgeting exercise or structure later. Again, all I ask is an hour of intention into budgeting tomorrow. And if other stuff comes up during that time we can figure out if the agreements need to be adjusted at all. I just want you to know that *my* concern is just that we have enough money to live life well and I want to make sure that we have enough of it both now and later to live the life that we want. Which, fmrn (for me right now), means spending conservatively in case there are big purchases we want to make in the future, so that we have the money to purchase them. And also just making sure that we always have enough to take care of our body, buy food and water and toiletries and housing/shelter if we need. And historically we have put a lot into estimating how much the minimum threshold is for all of that, but our quality of life is starting to increase and so that's just why I think the budget would be helpful.

I mean that was a lot and thank you for sharing and I'm glad you got it out. I don't really want to read it right now. Maybe we can revisit it tomorrow and make an intention doc for budgeting before we really get to crunching the numbers. What I want to do right now is provision the GPUs because that is required in order to get the repo working and it is within the budget. And the simplicity of that logic is also why the budget is helpful. In the budget => we can spend the money without worry. If you want to frame/analyze it, all the worry and calculation gets condensed into a single moment of higher context and abstraction, and the outcomes of that are made actionable with budgets. And the budgets can be changed but there is a process for that. So in short, now I want to provision the GPUs.

How much time is on the food? Nine minutes. Alright let's try to provision before the food is done. Which requires changing the settings. I need to configure a public key.

### Decision point: Install & Authorize RunPod Inc. for Github
Ok first, why are we here?

Well, maybe we'll need to give it Github access later.

Maybe, but we don't know that right now. We don't know what we would need it for and we have distracted from the primary objective. I think we should go back to the settings page or wherever.

Sorry. Are you mad at me?

I understand that you thought it would be a good idea but if we hadn't paused at this decision point it might've taken a long time and we wouldn't have gotten it done. I don't know if I'm mad at you. I think I'm frustrated because I believe this kind of thing happens a lot. I know you thought it was a good idea and I know you're only trying to help. I don't want you to stop having and advocating for these ideas. I do want us, in general to stop just making decisions without considering all the feelings.

Yeah but that's not sustainable. Sometimes things just need to happen and you've gotta move fast. Something something in stress you just execute the patterns you're used to.

Somehow I feel like that is a tangential point. Deciding to click on that button was a decision and it was in service of something other than what we had planned. Yes, it was related to the larger project that is part of our plan, but it was still a dis-traction. You do a couple of those in a row and all of a sudden you are working on something completely different than you had originally intended.

... Ok we are done eating. This is what I had initially intended to do for the day but there is still a lot to do. It seems like if we keep moving at this pace it'll never get done. Really wanting to just brainrot away the feelings right now. Or jack off or something.

If I just do the settings thing then we can get the GPUs provisioned and then we can at least move on to the next step. I know there are unresolved things going on right now but if I just do it we might feel better.

I am hot. I need to go to the bathroom. My posture is bad. It feels like there is too much to process.

Working on the ssh thing. I think we should use our j1ng3r public key for the time being. Maybe in the future it would be worth switching it to make sure it's associated with stopthrowingrocks but for the time being we just need it to work. Alright both the j1ng3r.pub, id_rsa.pub, and id_ed25519.pub keys are the same so I am using that one. I am just pasting the contents into the Public Key textbox in the Runpod Settings and hoping that that works.

Next: do we need a Jupyter notebook? I mean why not right. The README doesn't say anything about it but I'm sure it doesn't add any overhead.

Ok there is a template. I don't know what that means and I don't know if it's important but it looks like it might be important. Does the README say anything about it? ... This seems to be the relevant information from the README.
```bash
pip install --upgrade pip
pip install transformers datasets timeout_decorator wandb matplotlib pynvml neptune
pip install --upgrade "filelock>=3.12"
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```
The setting in the pod config is: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` which looks like a Docker image name. I am once again very scared of everything going on. I don't really know Docker. I have a vague idea of what it is and how it works. But it's not just that. Everything is new and unfamiliar. How do those wheels work? Can I use just the default image? It seems like the torch version in the Dockerfile name is different than the one the README asks us to install.

The great thing about having a budget is we can just try it and maybe everything works. [Would we know if everything was working correctly or is it possible that things would fail silently and the code would execute but it would just be worse than doing it right now?] I mean yeah that's possible. So why do I think it might work out fine? Well I'm guessing that the only things that the Dockerfile does are install various packages. Can we just go look at the Dockerfile? My reason for wanting to look at the Dockerfile is
1. The default path would have us be using this Dockerfile anyways, so any learnings would not be completely wasted.
2. In reading the Dockerfile and understanding how it works, we might better believe that using it is completely fine and will be compatible with what we need it for.
3. Maybe we learn that actually the Dockerfile is not needed at all, and the repo gives us everything we need. Honestly that is the preferred option if we can make it work.

But we kinda don't have a lot of time. What if we just use this Dockerfile and it works and it's fine and we can move on? I mean if it doesn't work we can always fix the problem then and we would actually be equipped with more information to solve the problem.

What's the worst thing that could happen? We provision it for an hour and mess it up and waste $3.20 off the budget. [Should we start with less GPUs at first?] While that would save us a bit of money, either way it won't really impact our odds of exceeding the budget that much and it adds more ways for things to go wrong. Therefore, I don't think we should do it.

Looks like if we click the "Change Template" button then we are taken to a page where we have options to select other configs, including one that says "pytorch:2.4.0" which agrees with the README in torch version. There is also another option which looks like raw Ubuntu. The concern with the pytorch one is that it will have too much stuff, and the concern with the Ubuntu one is that it won't have enough stuff. Aren't we glad that we didn't spend the time reading the Dockerfile? I think for now given my experience with Linux and how hard it is to get the stuff you don't have if you need it, I would rather risk having a bit too much stuff. Seems just easier to handle, but if we really run into problems then we can always just switch the template. ... I am giving the website $10 right now in case I leave it on and forget about it. That way we still have plenty of money before we hit the $120 budget maximum. ... Aaa the cost went up to $3.21/hr! I didn't know the cost could increase! I want to try SSHing into it and then I'll close it so I can play among us.

There are three options to ssh into it. There is SSH, SSH over exposed TCP, and Web Terminal. Web Terminal seems easy but it also seems fake easy. The worst case is I use the wrong SSH command and some hacker who is watching just completely snipes it and I lose $10. Well not just that, what if my private key gets leaked and all of a sudden all of my stuff is now insecure? I think this is important so I want to make sure I'm using the command correctly. But that's not happening now though. Let's shut down the instances and go play among us. I don't think I'll terminate the pod though because the storage is really cheap. Actually nah I didn't suffer any consequence (that I know of) from opening and closing it so I'll just close it. So we'll just recreate the pod from scratch once among us is over.

... So Among Us is completed. It's about 10pm so it took about 3 hours. We have work left to do to finish this. I think if we can do this now that is good news for our ability to do work in the future. Something self trust.

I mean, yeah. There is always a reason why we get distracted so if we're all trying to work together then we can discuss any motion to distract and deal with it in a healthier way. I guess there can always be feelings that don't make themselves linguistically interpretable but that's fine. Moving on, we have a lot of stuff to do.

So SSH. Well, I am also hot and the screen door fell out and the window is open and if I don't close the screen door then mosquitoes might fly in. Also I didn't get to work out or play piano today which is unfortunate. Let's try and make time for them tomorrow, and let's close the screen door first just so that's resolved. ... Screen door is closed. It still needs a more permanent fix. Other concerns: food and sleep, also acne and shower. [I want to jack off.] I don't feel very good and want something to just make it go away. I feel bad and I want to feel better. Will jacking off make me feel better? I can fix the hungry problem right now but I can't really fix the heat problem.

This is all just a lot. I feel overwhelmed maybe. What if I can't get this done tonight? When would I go to sleep?

I vote we just get the SSH thing done. The last time we got the instance running that was very good for us. It proved that a lot of things just worked and allowed us to move to the next stage. I don't know about the food thing but for the sleep thing I vote we revisit at 11pm.

Let's like put something in the oven to cook while we do other stuff. Take off the shirt and maybe put a towel on or something to get cooler. ... Showered and samosas in the oven. I would really like to put on deodorant but idk where it is. ... Deodorant found and applied. I'm glad I didn't have to go to the store to get some but I hope that I eventually (like tomorrow) find the other one because I don't like the smell of this one as much. It's just unfamiliar.

Ok now what. The food is in the oven and I am showered and a bit cooler. I am hungry but the food is coming. I think the best thing I can do right now is keep working on the SSH thing. ... Restarting the pod because I closed it last time. Enabling a jupyter notebook because it probably won't cause problems and it might be nice, who knows. ... It seems the main difference between the two is that the first command does not support SCP and SFTP and the second command does.
> SCP (secure copy) is a command-line utility that allows you to securely copy files and directories between two locations. https://linuxize.com/post/how-to-use-scp-command-to-securely-transfer-files/
> Secure File Transfer Protocol (SFTP), is a network protocol that provides file access, file transfer, and file management over any reliable data stream. https://en.wikipedia.org/wiki/SSH_File_Transfer_Protocol
Basically, the second command should be used if I anticipate needing to copy files over and the first command should be used otherwise.
Actually is there any reason not to use the second one? Maybe it's less secure or something. I feel like we should just use the second one just in case files do need to be transferred. Yes, we could've checked the README in case we do need to copy files but like what if there are commands that we don't know what they do? And yes maybe we won't encounter such commands, but I feel like the only difference in strategy we would take as a result is if we don't need to copy files but like we might still be worried as a result and use the other one anyways.

Don't assume that we will feel a certain way or end an interaction incompletely right now. Though I guess I do also see what you're saying. Well I need to take the samosas out anyways. ... Samosas are good but hot but good.

SSH'd in, and we're in! Nice! Let's keep going through the README. ... Interesting, `sudo` doesn't work. It says `-bash: sudo: command not found`. I'm logged in as root, maybe sudo doesn't exist when you're root? That's kind of weird, I would expect it to be idempotent.

Alright so after looking it up we can just ignore all the `sudo`s. In installing all the system packages it looked like some things might've gone wrong but at this point it's getting late and we just need to see if all the other steps work. This one seems like it has a good chance of working.

Both the README and the terminal are recommending making a virtual python environment first. Sure, why not. ... That wasn't that bad, I think we just made a sub-terminal and now the rest of the commands should just work fine in it. ... Great, no warning.

### Indentation Error
On trying to run the smoke test, it errored. I feel like I'm close to wanting to call it a night.
- `FutureWarning: The module torch.distributed.launch is deprecated and will be removed in future. Use torchrun.` I guess good to know but probably not the most pressing issue right now.
- Something about LOCAL_RANK and OMP_NUM_THREADS.
- Ok sorry I went and checked #lang and then the discord meme channels. I think I'm getting tired and we should call it a night soon.
- A bunch of `IndentationError: unexpected indent \n File "/zero_order_rnn/distributed_rge.py", line 720`. That seems like something we could look at the code to fix.

After looking at the code in my local repo, I immediately found the error. However, right now the GPUs have cloned their version of the repo. I see a few ways to fix this.
- Directly edit the files on the instance.
- Link it to my repo, commit on my repo, and then pull changes on the instance.
- Somehow set up like a more ergonomic serverless-y option. I remember seeing something earlier about that. Right, Instant Clusters. I could look into that.

I think if the third option doesn't work out we start with the first option and then migrate to the second option once/if the changes we need to make are more complex than just changing the indentation.

However, these are not things that will happen tonight. In the morning. And if we don't finish getting everything working by the time we talk with him, we'll just report on our progress so far.

The water in my bottle kind of tastes like soap because I washed it. :/

... It is the following morning and I do not feel good. Fancy that. Pain on the top of the head. The last time this happened I needed to go out and walk around outside and I was using the concept of the Crown Chakra as my guide/framework for thinking about what to do. I don't want to just hop right into this because I anticipate that if I do the pain would increase/the sensation would get more intense and I wouldn't be able to focus on the work. (Maybe pain is just sensation? Maybe Teal Swan has a video about this? I can look it up later if I want.)

Well, now what. I don't feel like anything really got resolved and I want to work on the Capstone but I don't know what the pain wants and so I can't resolve it.

iisddujjvcjhisinjvd ioh oijokmvch apzihvbre lskcvhjiseu cjoisjsd p hdjivi soijsdifjpwe3,jcuhwiod

That didn't work. :/ I thought that if I randomly typed letters with my eyes closed that maybe something intelligible would come out. Alright, let's try staring at the ceiling.

... So I am still tired and am liable to make more mistakes than usual in this condition. So let's just take it slow and not get into situations where we don't have a solid plan on what to do. We can nap if it would be helpful or lie down and contemplate plans if we encounter a hard problem. However, getting information is also important so that we can plan correctly. Right now, we should look up the best way to edit files on remote linux and then go do that. ... It seems like our two best options are something like vim or emacs and a file manager. Given that right now we just want to remove a single space, let's just use vim.

Also last night we left the pod running which probably cost us about $8. Just reporting the information for now, we don't need to make any decisions or adjustments yet. We are weighing the administrative overhead of remembering to turn off the pod when we aren't going to use it for a while versus the monetary cost against the budget.

So flashrnn. I think what's going on is that we are cloning the repo locally, then installing the contents of the repo to pip locally. That means we wouldn't need to install it in any particular location and we can now use flashrnn in python files. But why not just run `pip install flashrnn`? ... Apparently the `-e` flag installs it in "editable mode". Not sure what that means. Are the contents of the flashrnn directory ever modified in development?

## Running the Smoke Test
Ran into an error:
```
Retrying in 1s [Retry 1/5].
HTTP Error 429 thrown while requesting HEAD https://huggingface.co/datasets/haritzpuerto/the_pile_00_OpenWebText2/resolve/cd746a83b3095458b1116b75bb9a840850e30b56/data/train-00007-of-00009.parquet
Retrying in 2s [Retry 2/5].
HTTP Error 429 thrown while requesting HEAD https://huggingface.co/datasets/haritzpuerto/the_pile_00_OpenWebText2/resolve/cd746a83b3095458b1116b75bb9a840850e30b56/data/train-00007-of-00009.parquet
Retrying in 4s [Retry 3/5].
HTTP Error 429 thrown while requesting HEAD https://huggingface.co/datasets/haritzpuerto/the_pile_00_OpenWebText2/resolve/cd746a83b3095458b1116b75bb9a840850e30b56/data/train-00007-of-00009.parquet
Retrying in 8s [Retry 4/5].
HTTP Error 429 thrown while requesting HEAD https://huggingface.co/datasets/haritzpuerto/the_pile_00_OpenWebText2/resolve/cd746a83b3095458b1116b75bb9a840850e30b56/data/train-00007-of-00009.parquet
Retrying in 8s [Retry 5/5].
HTTP Error 429 thrown while requesting HEAD https://huggingface.co/datasets/haritzpuerto/the_pile_00_OpenWebText2/resolve/cd746a83b3095458b1116b75bb9a840850e30b56/data/train-00007-of-00009.parquet
Hugging face issue...
An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.
HTTP Error 429 thrown while requesting HEAD https://huggingface.co/datasets/haritzpuerto/the_pile_00_OpenWebText2/resolve/cd746a83b3095458b1116b75bb9a840850e30b56/data/train-00007-of-00009.parquet
Retrying in 1s [Retry 1/5].
HTTP Error 429 thrown while requesting HEAD https://huggingface.co/datasets/haritzpuerto/the_pile_00_OpenWebText2/resolve/cd746a83b3095458b1116b75bb9a840850e30b56/data/train-00007-of-00009.parquet
Retrying in 2s [Retry 2/5].
HTTP Error 429 thrown while requesting HEAD https://huggingface.co/datasets/haritzpuerto/the_pile_00_OpenWebText2/resolve/cd746a83b3095458b1116b75bb9a840850e30b56/data/train-00007-of-00009.parquet
Retrying in 4s [Retry 3/5].
HTTP Error 429 thrown while requesting HEAD https://huggingface.co/datasets/haritzpuerto/the_pile_00_OpenWebText2/resolve/main/README.md
Retrying in 1s [Retry 1/5].
```
So it seems like it was trying to fetch a dataset file, ran into HTTP errors, and ran out of retries. I would think if this happened the whole thing would fail but apparently not because it seems like it just tried getting the file again and after 3 retries it moved on so maybe it was successful??? Going to assume it worked but if other things go wrong maybe this will be helpful.

The logs are stalled after a bunch of "Got the OWT dataset!" logs, but the GPU utilization is still at 100%, and the VRAM utilization is up to 55% and the Container Disk Utilization is up to 62%. I think if the memory gets all used up that's fine but if the Disk gets used up that's bad. Also of note, the CPU Utilization is at 12% and the Pod Volume Utilization is at 40%. I do not entirely know what all these terms mean.

... Errors are happening in the terminal. Lemme poop and then look at them. ... It starts with something that says "Watchdog caught collective operation timeout." Let's look it up and see what's going on. ... Idk which link to start with. Maybe the Pytorch one?

Kind of ignoring the article for now. The README mentions something about InfiniBand. Let's check if our GPUs have them. ... Don't see the word anywhere on the Pod description. Let's look up what Infiniband is. ... Reading this article, I am thinking about stuff not directly related to this project. The trigger was "In practical terms, faster networking means your GPUs spend less time waiting on each other and more time crunching data." and I thought about how neurons in the brain are probably pretty optimized, and how the optimization process is typically ascribed to evolution, but in the case of GPU optimization, humans are the ones optimizing their performance. And it called to mind that there might be a completely different world of entities with internal structure comparable to that of human society, responsible for the optimization of things like our bodies and their function. This would be a world of spirits or memes or something idk. [But surely the brain is doing real computation! Surely computation is physical and connected to things like the Landauer limit.] And that calls to mind Celeste's concept of "measure" which I still don't have a good grounding on. ... Lot's of other cool things to do but right now I should focus on preparing for the call at 1:45pm.

> When you deploy an Instant Cluster on Runpod, the nodes are connected via low-latency datacenter interconnects – in many cases using InfiniBand or similar technology – to ensure efficient scaling.

AaaaaaAaAAAAa

So it might not be using infiniband which might be a problem. idk why it would be a problem though. Let's read that other article.

... Alright I am just trying the smoke test again with much smaller model size hyperparameters. ... Didn't work, now the error is that CUDA is unavailable. This is frustrating. Their repo is not set up to be approachable. Really? You're going to train a 7B parameter model as the test to make sure things are running? I really want to do something else and just talk to him and say "I read the paper and thought it was really cool, but I struggled a lot to get the actual repository working." What if that looks bad on me though? Am I expected to be able to work through problems like this? Basically, do I have the right to get a little mad and ask for help here? I mean if I didn't have help (and I was sufficiently motivated to do this on my own) I would just need to figure it out on my own but also right now there's a time pressure.

I mean if I imagine this as training for a future job then I should be able to work through it, but even then I would probably slack my coworker something like "I'm getting a lot of errors trying to get this set up, is that anticipated?" At least in that case the information gets out that I had problems, so even if I'm able to resolve them on my own my teammates can also do a bit to make their stuff more approachable, or set the expectation that "we're scrappy and so yeah you need to just figure out how to make it work on your machine."

Ok, how would we fix this? Let's read through the distributed_rge.py file and see what's going on.
