// Mounted hook regression: an outer snapshot can advance without changing region identity.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
const region = { id: "r", block_ids: ["b"], mapping_status: "complete", body_status: "linked" };
const block = { block_id: "b", chapter_id: "c", section_id: "s", task_unit_id: "u", content: "Current", deleted: false };
const initial = { revision: 1, source_token: "source", structure_version: "v1", regions: [region], blocks: [block],
  block_page: { total: 1, offset: 0, limit: 20 } };
try {
  for (const change of [{ revision: 2 }, { source_token: "replacement" }, { structure_version: "v2" }]) {
    for (const phase of ["refresh", "navigation", "content", "verification"]) {
      const page = await browser.newPage();
      const errors = [];
      page.on("pageerror", error => errors.push(error.message));
      let held = null, hold = false;
      await page.route("**/api/**", route => {
        if (new URL(route.request().url()).pathname.endsWith("/reading-content")) {
          if (hold && (phase !== "verification" || new URL(route.request().url()).searchParams.has("block_id"))) {
            held = route; return;
          }
          return route.fulfill({ json: initial });
        }
        return route.fulfill({ json: { documents: [] } });
      });
      await page.goto(process.env.UI_URL || "http://127.0.0.1:5176");
      if (phase === "refresh" && "revision" in change) {
        const matches = await page.evaluate(async () => {
          const { matchesReviewDisplayContent: matches } = await import("/src/features/reader-content/sourceReviewModel.ts");
          const cases = [
            ["same", "same", undefined, true],
            ["alpha beta", "alpha\nbeta", "ocr_whitespace_v1", true],
            ["alpha\n\nbeta", "alpha\n\nbeta", "ocr_whitespace_v1", true],
            ["\u4e2d\u6587ABC", "\u4e2d \u6587\nABC", "ocr_whitespace_v1", true],
            ["AB,.", "\uff21\uff22, .", "ocr_whitespace_v1", true],
            ["alpha beta", "alpha\nbeta", undefined, false],
            ["alpha beta", "alpha\nbeta", "unknown", false],
            ["old cached text", "current text", "ocr_whitespace_v1", false],
            ["ab", "a b", "ocr_whitespace_v1", false],
          ];
          return cases.map(([displayed, raw, kind, expected]) => matches(displayed, raw, kind) === expected);
        });
        assert.ok(matches.every(Boolean), "Only the explicit backend display projection may differ from raw text");
        console.log("PASS 9 display-normalization comparison cases");
      }
      await page.evaluate(async ({ snapshot, phase }) => {
        const { default: React } = await import("/node_modules/.vite/deps/react.js");
        const { default: ReactDOMClient } = await import("/node_modules/.vite/deps/react-dom_client.js");
        const { default: ReactDOM } = await import("/node_modules/.vite/deps/react-dom.js");
        const { useSourceReviewBlocks } = await import("/src/features/reader-content/useSourceReviewBlocks.ts");
        const { useReaderContentController } = await import("/src/features/reader-content/controller.ts");
        const container = document.createElement("div"); document.body.append(container);
        const root = ReactDOMClient.createRoot(container);
        const test = window.reviewTest = { published: [], navigated: [], snapshot };
        const chapter = { chapter_id: "c", sections: [{ section_id: "s", task_units: [{ unit_id: "u" }] }] };
        const layout = { document_id: "Book", chapters: [chapter] };
        const service = { fetchTaskUnitContents: async () => {
          if (phase === "content") await new Promise(resolve => { test.releaseContent = resolve; });
          return [{ document_id: "Book", chapter_id: "c", section_id: "s", task_unit_id: "u",
            content_blocks: [{ block_id: "b", block_type: "paragraph", content: "Current" }] }];
        } };
        function Probe() {
          test.reader = useReaderContentController({ docName: "Book", layout, service });
          test.hook = useSourceReviewBlocks("Book", "r", 0, test.snapshot,
            next => test.published.push(next), async (block, snapshot, signal, assertCurrent) => {
              if (phase === "content" || phase === "verification") {
                await test.reader.navigateToBlock(block, snapshot, signal, assertCurrent);
              }
              test.navigated.push(block.block_id);
            });
          return null;
        }
        test.update = next => { test.snapshot = next; ReactDOM.flushSync(() => root.render(React.createElement(Probe))); };
        test.update(snapshot);
        test.dispose = () => root.unmount();
      }, { snapshot: initial, phase });
      await page.waitForFunction(() => window.reviewTest?.hook.busy === false);
      hold = phase !== "content";
      await page.evaluate(phase => {
        window.reviewTest.published = [];
        if (phase !== "refresh") window.reviewTest.hook.navigate("b"); else window.reviewTest.hook.reload();
      }, phase);
      if (phase === "content") await page.waitForFunction(() => Boolean(window.reviewTest.releaseContent));
      else while (!held) await page.waitForTimeout(10);
      await page.evaluate(snapshot => window.reviewTest.update(snapshot), { ...initial, ...change });
      if (phase === "content") await page.evaluate(() => window.reviewTest.releaseContent());
      else await held.fulfill({ json: initial });
      await page.waitForFunction(() => !window.reviewTest.hook.busy && window.reviewTest.hook.error);
      const result = await page.evaluate(() => ({ error: window.reviewTest.hook.error,
        state: window.reviewTest.hook.state, published: window.reviewTest.published, navigated: window.reviewTest.navigated,
        blocks: window.reviewTest.reader.contentBlocks, target: window.reviewTest.reader.navigationTarget }));
      assert.equal(result.error, "Content changed. Refresh text review.");
      if (phase === "refresh" || phase === "navigation") {
        assert.equal(result.state, null);
        assert.deepEqual(result.published, []);
      }
      assert.deepEqual(result.navigated, []);
      assert.deepEqual(result.blocks, []);
      assert.equal(result.target, null);
      assert.deepEqual(errors, []);
      await page.evaluate(() => window.reviewTest.dispose());
      await page.close();
      console.log("PASS latest snapshot " + Object.keys(change)[0] + " during " + phase);
    }
  }
} finally { await browser.close(); }
