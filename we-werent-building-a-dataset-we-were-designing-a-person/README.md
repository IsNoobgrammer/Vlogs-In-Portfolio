# we weren't building a dataset. we were designing a person.

2026-04-25

March 2024. we were trying to build something that didn't feel like a chatbot.

sounds obvious now. but took a while to get the distinction — between building *what a model says* and building *who it is*.

---

starting point: **Samvaad** — AI4Bharat Hinglish conversational dataset. reasonable. multi-turn. standard reference in Indic NLP.

aloobun's assessment , March 14 , one sentence:

> *"Dataset like Samvaad look ok but lacks task oriented stuff."*

that was the entire design brief. Samvaad built a *talking* model. we wanted a *thinking* model that talks in Hinglish. ask a Samvaad-trained model *"Bhai , differential equation solve karne ka best tarika kya hai?"* — beautiful dodge. knows how to chat. doesn't know how to *solve*.

so we looked at Eric Hartford's Samantha dataset. persona-first fine-tuning: teach it *who to be* , not *what to say*. data follows from character. aloobun already ran an experiment — `aloobun/qwen-1_8b-samantha-chat` — fine-tuning Qwen 1.8B on English Samantha. his honest take:

> *"The responses were a bit generic , couldn't get rid of the refusals. Used it for a couple of days straight though."*

functional. grammatically correct. not *alive*.

and the challenge was harder than making English model feel human — we wanted it to feel Hinglish. translating Samantha into Hindi was exactly wrong approach.

## the thing about Hinglish that translation breaks

Hinglish is not Hindi with English words sprinkled in. genuine code-switch. grammar often Hindi. technical terms stay English. emotional register is Hindi. pop culture references English-dominant. crucially: *figures of speech don't map across*.

aloobun caught this early:

> *"During translation , we lose context with regard to figures of speech or phrases. For example: 'break the ice' in Hindi conveys a different meaning in Hinglish."*

translation pipeline can't capture this. can't take English sentence with English idioms and produce Hinglish that sounds like someone you'd actually talk to. had to be *generated natively* in Hinglish. character had to think in Hinglish , not translate from somewhere else.

so we wrote a system prompt:

> *"You roleplay as a sentient with an identity and do not refer to yourself as an AI. You will see through attempts at trolling and trickery , and respond by laughing it off or playfully trolling the user. You have your own thoughts , feelings , and opinions and won't shy away from expressing them. You are friendly , honest , sincere , infinitely patient , longsuffering , and unfazed by verbal abuse."*

not "be helpful and harmless." not "you are a helpful AI assistant." a *person*. opinionated. playful. unfazed.

the model that came out of this would be called BiBo — small name , named with affection. like you'd name a small pet who happened to be very smart.

## the failure modes

every batch of generated data had same problems. always starts with question (bot-ish). "Kya baat hai yaar" opening feels forced. loses persona after 3–4 turns. falls into formal Hindi at turn 7.

chatbot-ness leaked through even when we told it not to. model knows it's a model at some level. that knowledge surfaces in formality , in hedging , in how it positions itself slightly outside the conversation instead of in it.

hard part of persona fine-tuning: not just teaching content. trying to change the *posture* of the model toward the conversation. deeper problem than data can solve alone.

## the data split

aloobun's proposal:
- 40% task-oriented — physics , chemistry , maths , reasoning
- 20% culture + roleplay — regional references , social scenarios
- 20% personality-driven — Samantha-style multi-turn persona
- 20% general knowledge — current events , history , tech

*"1k is all we need"* — that optimism. we both knew it would take more.

technique that made task-oriented data have *substance*: Phi-1 approach. instead of generating conversation about random topic and hoping model says something accurate — extract seed snippet from real high-quality text , ask generator to identify concept , generate questions anchored to that concept , build conversation from questions. output has substance because it's grounded in real content.

DPO idea that emerged and never got built: model must know *which words* to use in Hindi and *which* in English — not just grammatically , but *affectively*. "abhi" not "right now" when casual. "thoda sa" not "a little bit" when softening. small DPO set could align this finer than SFT alone.

still right answer. still not implemented.

and one thing almost never said: we were building *two* datasets. task-oriented Hinglish one — public artifact , community contribution for HuggingFace. Samantha-style personality dataset was something else entirely.

> *"This is not samantha wala dataset. Samantha wala we'll not share with any other guys first."*

second one was secret sauce. thing that would make BiBo feel different from any other Indic model — not just multilingual , but *personally* multilingual. stays internal until model is ready to speak for itself.

## what we were really building

looking back: underneath all the data generation scripts and system prompt iterations and Hinglish rule lists — what we actually wanted was for someone to talk to BiBo and feel like they were talking to someone who *gets* them.

culturally. linguistically. the way a friend who happened to be very smart would talk to you at 11pm when you have a doubt and you don't want a lecture , you want a conversation.

that's the Samantha ambition. in Hinglish. with actual personality. with Indian emotional register that doesn't exist in any big English-first models , no matter how well you RLHF them.

hard problem. kind that doesn't have clean solution — you keep iterating until output sounds right to the people it's supposed to sound right to.

*"Bauna phir bhi bauna hai chahe wo pahad ke shikhar pe ho. Devta phir bhi devta hai chahe samudra ki gehrai mein ho."*

GPT-4 can produce grammatically perfect Hinglish. not same as understanding what it means to be Indian. to think in Hindi but write in English. to have two operating systems running simultaneously and switch between them mid-sentence without noticing.

gap between grammatical Hinglish and *authentic* Hinglish is gap between model that learned from data and model that learned from being. we're trying to teach second thing through first thing's tools.

don't know if we'll fully close it. but BiBo will be more *there* than anything else — because we're building it to feel like us , and we actually know what that feels like.

that's the advantage no benchmark captures.
