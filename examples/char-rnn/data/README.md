# char-rnn data

`essay.txt` is committed: about 1.8 kB of text about this model, small enough to train in a
minute and large enough to watch the loss leave uniform noise. `train.ts` reads it by default.

`tiny-shakespeare.txt` is not committed, because it is 1.1 MB. Fetch it with:

```sh
pnpm fetch:corpus
```

It is the Tiny Shakespeare corpus from the original
[char-rnn](https://github.com/karpathy/char-rnn) repository, which is public domain.
