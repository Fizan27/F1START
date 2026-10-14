# Deploying F1-EQUALIZER to Vercel

Everything here is free except the assistant's API calls, which are billed by
Anthropic to whoever owns the API key. The site works without a key; only
the Pit Wall panel is switched off.

## 1. Put the site on Vercel (about 3 minutes)

1. Go to **vercel.com** and click **Sign Up** (or **Log In**). Choose
   **Continue with GitHub** and approve.
2. On your dashboard click **Add New…** then **Project**.
3. Under **Import Git Repository**, find the repository (currently named
   **F1START**) and click **Import**. If it is not listed, click **Adjust
   GitHub App Permissions**, tick the repository, save, and come back.
4. On the **Configure Project** screen:
   - **Root Directory**: click **Edit**, choose **web**, click **Continue**.
     This matters: the website lives in the `web` folder.
   - **Framework Preset** should now say **Next.js**. Leave the build and
     output settings as they are.
5. Click **Deploy**. After a minute or two you get a link like
   `https://f1start.vercel.app`. Open it: every page should work, and the
   **Ask Pit Wall** button should say the assistant is not available (no key
   yet).

## 2. Switch on the Pit Wall assistant (optional, costs money)

### Get an API key and set a spend limit first

1. Go to **console.anthropic.com** and sign in (or create an account).
2. Open **Settings**, then **Billing**, and add credit if needed.
3. Still in **Settings**, open **Limits** and set a **monthly spend limit**
   you are comfortable with (for example $5). This is the real protection
   against a public link running up a bill; the site's own rate limits are
   only a brake.
4. Open **API Keys**, click **Create Key**, give it a name such as
   `f1-equalizer-vercel`, and copy the key. You will not see it again.

Never paste the key into any file in the repository. It is public.

### Add the key to Vercel

1. In Vercel, open the project, then **Settings**, then **Environment
   Variables**.
2. Add:
   - **Key**: `ANTHROPIC_API_KEY`
   - **Value**: the key you copied
   - **Environments**: tick **Production** (and **Preview** if you want it
     on preview links too)
3. Click **Save**.
4. Environment variables only apply to new deployments. Open the
   **Deployments** tab, click the **…** menu on the latest deployment and
   choose **Redeploy**.

### Optional settings (same Environment Variables screen)

| Key | What it does | Default |
|---|---|---|
| `PIT_WALL_MODEL` | Which Claude model answers. `claude-haiku-5-5` costs about a fortieth as much per token as the default, at some loss of quality | `claude-opus-5-5` |
| `PIT_WALL_PER_10_MIN` | Questions one visitor may ask per 10 minutes | 5 |
| `PIT_WALL_PER_DAY` | Questions one visitor may ask per day | 20 |
| `PIT_WALL_SITE_PER_DAY` | Questions the whole site answers per day (per running server copy) | 100 |

## 3. Check it works

1. Open the live site and click **Ask Pit Wall**.
2. Click the example question "Would Senna beat Verstappen at Monaco in equal
   cars?". You should get a short answer with percentages and a range, a line
   saying which tools were used, and a reminder that it is a model estimate.
3. If it says the assistant is unavailable, the key is missing or the site was
   not redeployed after adding it. If it says it could not answer, check the
   **Logs** tab of the deployment in Vercel.

## 4. After deploying

- Put the live link at the top of `README.md` (replace "not deployed yet").
- To measure the assistant's evaluation score on your own machine:
  `cd web`, then `$env:ANTHROPIC_API_KEY = "your key"` and `npm run eval`
  (about 50 questions, so it costs a little; with the default model, roughly
  a few US cents per question). Add the four scores to the README.
- Every push to the `main` branch on GitHub redeploys the site automatically.
