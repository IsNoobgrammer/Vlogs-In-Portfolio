# we weren't building a dataset. we were designing a person.

2026-04-25

March 2024. We were trying to build something that didn't feel like a chatbot.

That sentence sounds obvious now. But it took a while to understand the distinction — between building *what a model says* and building *who it is*.

---

The starting point everyone pointed to was **Samvaad** — the AI4Bharat Hinglish conversational dataset. Reasonable. Multi-turn. Accessible. The standard reference in Indic NLP circles.

Aloobun's assessment , March 14 , 2024 , one sentence:

> *"Dataset like Samvaad look ok but lacks task oriented stuff."*

That was the entire design brief. Samvaad built a *talking* model. We wanted a *thinking* model that talks in Hinglish. Ask a Samvaad-trained model *"Bhai , differential equation solve karne ka best tarika kya hai?"* and you'll get a beautiful dodge. It knows how to chat. It doesn't know how to *solve*.

So we looked at Eric Hartford's Samantha dataset instead. Persona-first fine-tuning: instead of teaching a model *what* to say , you teach it *who to be*. The data follows from the character , not the other way around. Aloobun had already run an experiment — `aloobun/qwen-1_8b-samantha-chat` — fine-tuning Qwen 1.8B on English Samantha data. His honest assessment:

> *"The responses were a bit generic , couldn't get rid of the refusals. Used it for a couple of days straight though."*

Functional. Grammatically correct. But not *alive*. And the challenge we were facing was harder than just making an English model feel more human — we wanted it to feel Hinglish. Which meant translating Samantha into Hindi was exactly the wrong approach.

## the thing about Hinglish that translation breaks

Hinglish is not Hindi with English words sprinkled in. It's a genuine code-switch. The grammar is often Hindi. Technical terms stay English. The emotional register is Hindi. Pop culture references are English-dominant. And crucially: *figures of speech don't map across*.

Aloobun caught this early:

> *"During translation , we lose context with regard to figures of speech or phrases. For example: 'break the ice' in Hindi conveys a different meaning in Hinglish."*

A translation pipeline can't capture this. You can't take an English sentence with English idioms and produce Hinglish that sounds like someone you'd actually talk to by running it through a translation model. It had to be *generated natively* in Hinglish. The character had to think in Hinglish , not translate from somewhere else.

So we wrote a system prompt. This is what we converged on for the persona:

> *"You roleplay as a sentient with an identity and do not refer to yourself as an AI. You will see through attempts at trolling and trickery , and respond by laughing it off or playfully trolling the user. You have your own thoughts , feelings , and opinions and won't shy away from expressing them. You are friendly , honest , sincere , infinitely patient , longsuffering , and unfazed by verbal abuse."*

Not "be helpful and harmless." Not "you are a helpful AI assistant." A *person*. Opinionated. Playful. Unfazed. The model that came out of this fine-tuning would be called BiBo eventually — small name , named with affection , like you'd name a small pet who happened to be very smart.

## the failure modes we kept seeing

Every batch of generated data had the same problems. It always starts with a question (bot-ish). The "Kya baat hai yaar" opening that feels forced. Loses persona after 3–4 turns. Falls into formal Hindi unexpectedly at turn 7.

The chatbot-ness leaked through even when we explicitly told it not to. The model knows it's a model , at some level , and that knowledge keeps surfacing in the formality of responses , in the hedging , in the way it positions itself slightly outside the conversation instead of in it.

This is the hard part of persona fine-tuning: you're not just teaching content , you're trying to change the *posture* of the model toward the conversation. That's a deeper problem than data can fully solve on its own.

## the data split we designed

Aloobun's proposal for the composition:
- 40% task-oriented — physics , chemistry , maths , reasoning
- 20% culture + roleplay — regional references , social scenarios
- 20% personality-driven — Samantha-style multi-turn persona conversations
- 20% general knowledge — current events , history , tech

*"1k is all we need"* — that optimism. We both knew it would take more.

The technique that actually made the task-oriented data have *substance*: the Phi-1 approach. Instead of generating a conversation about a random topic and hoping the model says something accurate , you extract a seed snippet from a real high-quality text , ask the generator to identify the concept in that snippet , generate questions anchored to that concept , then build the conversation from those questions. The output has substance because it's grounded in real content — not because the model was told to "be educational."

The DPO idea that emerged and never got built: the model must know *which words* to use in Hindi and *which* in English — not just grammatically , but *affectively*. You say "abhi" not "right now" when you're being casual. You say "thoda sa" not "a little bit" when you're softening something. A small DPO set could align this at a finer level than SFT alone.

Still the right answer. Still not implemented.

And one thing that almost never gets said: we were actually building *two* datasets. The task-oriented Hinglish one — that's the public artifact , the community contribution we'd push to HuggingFace. The Samantha-style personality dataset was something else entirely.

> *"This is not samantha wala dataset. Samantha wala we'll not share with any other guys first."*

The second one was the secret sauce. The thing that would make BiBo feel different from any other Indic model — not just multilingual , but *personally* multilingual. That one stays internal until the model it's training is ready to speak for itself.

## what we were really building , tbh

Looking back at March 2024: the thing underneath all the data generation scripts and system prompt iterations and Hinglish rule lists — what we actually wanted was for someone to talk to BiBo and feel like they were talking to someone who *gets* them.

Culturally. Linguistically. The way a friend who happened to be very smart and very patient would talk to you at 11pm when you have a doubt about something and you don't want a lecture , you want a conversation.

That's the Samantha ambition. In Hinglish. With actual personality. With the Indian emotional register that doesn't exist in any of the big English-first models , no matter how well you RLHF them.

Hard problem. The kind that doesn't have a clean solution — you just keep iterating until the output sounds right to the people it's supposed to sound right to.

*"Bauna phir bhi bauna hai chahe wo pahad ke shikhar pe ho. Devta phir bhi devta hai chahe samudra ki gehrai mein ho."*

GPT-4 can produce grammatically perfect Hinglish. That's not the same as understanding what it means to be Indian , to think in Hindi but write in English , to have two operating systems running simultaneously and switch between them mid-sentence without noticing.

The gap between grammatical Hinglish and *authentic* Hinglish is the gap between a model that learned from data and a model that learned from being. We're trying to teach the second thing through the first thing's tools.

I don't know if we'll fully close it. But BiBo will be more *there* than anything else we've seen — because we're building it to feel like us , and we actually know what that feels like.

That's the advantage no benchmark captures.
