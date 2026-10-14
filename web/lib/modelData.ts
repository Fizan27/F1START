// lib/modelData.ts
//
// What it does: gathers the exported JSON files into one ModelData object for
//   the assistant's tools, on the server. They are imported (not fetched) so
//   that they are bundled with the server code and are always available.
// What it reads: public/data/*.json (written by export_web.py and
//   explain_ratings.py) and public/data/circuits/index.json.
// What it produces: MODEL_DATA.
// Which files use it: app/api/chat/route.ts and eval/run-eval.ts.

import circuits from "../public/data/circuits/index.json";
import details from "../public/data/details.json";
import drivers from "../public/data/drivers.json";
import explain from "../public/data/explain.json";
import h2h from "../public/data/h2h.json";
import records from "../public/data/records.json";
import settings from "../public/data/settings.json";
import type { ModelData } from "./tools";

export const MODEL_DATA = { drivers, h2h, details, records, explain, settings, circuits } as unknown as ModelData;
