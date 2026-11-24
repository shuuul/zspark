// Runs sparkDash's LlmProbe against a live engine and prints the snapshot.
//   node tools/sparkdash_probe.mjs [host] [port] [sleep_s]
// With sleep_s it polls, sends one repeat-image OCR request mid-way, sleeps,
// then polls again so counters that need an in-session delta (perStreamTps,
// generationTps, cachedPrefillTps) are exercised. Default 127.0.0.1 18888.
// SparkDash source is expected at /tmp/sparkDash or $SPARKDASH_DIR.
import { readFileSync } from "node:fs";

const dir = process.env.SPARKDASH_DIR ?? "/tmp/sparkDash";
const { LlmProbe } = await import(`${dir}/server/collectors/LlmProbe.js`);

const host = process.argv[2] ?? "127.0.0.1";
const port = Number(process.argv[3] ?? 18888);
const gap = Number(process.argv[4] ?? 0);
const probe = new LlmProbe({ lanIp: host, isLocal: host === "127.0.0.1" }, port);

const first = await probe.probe();
if (!gap) {
  console.log(JSON.stringify(first, null, 1));
  process.exit(0);
}
console.log(JSON.stringify({ first: pick(first) }, null, 1));

const png = readFileSync("zig/tests/fixtures/vision/shapes.png").toString("base64");
const body = JSON.stringify({
  model: first.modelId ?? "Unlimited-OCR-2",
  max_tokens: 48,
  messages: [{ role: "user", content: [
    { type: "image_url", image_url: { url: `data:image/png;base64,${png}` } },
    { type: "text", text: "<PAGE><image>Multi page parsing." } ] }],
});
await fetch(`http://${host}:${port}/v1/chat/completions`, {
  method: "POST", headers: { "Content-Type": "application/json" }, body,
}).then((r) => r.json()).then((r) => console.error("mid request:", JSON.stringify(r.usage)));

await new Promise((r) => setTimeout(r, gap * 1000));
const second = await probe.probe();
console.log(JSON.stringify({ second: pick(second) }, null, 1));

function pick(s) {
  const keys = ["backend","engineVersion","generationTps","prefillTps","perStreamTps",
    "perStreamPrefillTps","ttftSeconds","ttftP95Seconds","e2eP95Seconds","itlP95Seconds",
    "prefixCacheHitRate","cachedPrefillTps","uncachedPrefillTps","totalPromptTokens",
    "totalCachedTokens","totalOutputTokens","requestsRunning","requestsWaiting",
    "kvCacheUsage","preemptionsTotal","mtpAcceptanceRate","slotsActive","slotsTotal",
    "prefillActive","measurementValid"];
  return Object.fromEntries(keys.map((k) => [k, s[k] ?? null]));
}
