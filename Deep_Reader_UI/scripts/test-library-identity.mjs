// Run against a local Vite server; all API traffic is intercepted, no backend writes.
// PLAYWRIGHT_MODULE may point to an existing Playwright installation.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}),
});
const base = process.env.UI_URL || "http://127.0.0.1:5173";
const layout = (name) => ({ document_id: name, title: name, chapters: [{
  chapter_id: `${name}-c`, title: `${name} chapter`, sections: [{
    section_id: `${name}-s`, title: `${name} section`, task_units: [{ unit_id: `${name}-u` }],
  }],
}] });
try {
  for (const staleOptions of [false, true]) {
    const page = await browser.newPage();
    const errors = [];
    const contentRequests = [];
    const unexpected = [];
    let listCalls = 0;
    let slowReads = 0;
    let releaseSlow;
    let slowStarted;
    const slowSeen = new Promise((resolve) => { slowStarted = resolve; });
    page.on("pageerror", (error) => errors.push(error.message));
    await page.route("**/api/**", async (route) => {
      const path = new URL(route.request().url()).pathname;
      const body = route.request().postDataJSON();
      let data;
      if (path === "/api/documents" || path === "/api/documents/management") {
        listCalls++;
        const names = staleOptions && listCalls === 1 ? ["Old"] : ["New", "Slow", "Next"];
        data = { items: names.map((name) => ({ doc_name: name, title: name, source: "structured", preparation_status: "ready" })), total: names.length };
      } else if (path === "/api/documents/task-layout") {
        if (body.doc_name === "Slow" && ++slowReads > 1) {
          await new Promise((resolve) => { releaseSlow = resolve; slowStarted(); });
        }
        data = layout(body.doc_name);
      } else if (path.endsWith("/task-units/content")) {
        contentRequests.push({ path: decodeURIComponent(path), body });
        data = { contents: [] };
      } else {
        unexpected.push(path);
        return route.fulfill({ status: 500, json: { detail: "Unexpected API request" } });
      }
      await route.fulfill({ json: data });
    });
    await page.goto(base);
    const input = page.getByRole("combobox");
    if (staleOptions) {
      await input.click();
      await page.getByRole("option", { name: "Old", exact: true }).waitFor();
      await input.press("Escape");
    }
    async function openLibrary(name) {
      await page.getByRole("button", { name: "Library", exact: true }).click();
      await page.locator(".file-management-row").filter({ hasText: name })
        .getByRole("button", { name: "Open", exact: true }).click();
    }
    await openLibrary("New");
    await page.locator(".hierarchy-section-button").filter({ hasText: "New section" }).waitFor();
    assert.equal(await input.inputValue(), "New");
    await input.fill("uncommitted search");
    await input.press("Escape");
    const contentResponse = page.waitForResponse((r) => r.url().endsWith("/New/task-units/content"));
    await page.locator(".hierarchy-section-button").click();
    await contentResponse;
    assert.deepEqual(contentRequests.at(-1), {
      path: "/api/documents/New/task-units/content", body: { task_unit_ids: ["New-u"], segmented: true },
    });
    await openLibrary("Slow");
    await slowSeen;
    await openLibrary("Next");
    await page.locator(".hierarchy-section-button").filter({ hasText: "Next section" }).waitFor();
    const slowResponse = page.waitForResponse((r) => r.url().endsWith("/task-layout") && r.request().postDataJSON().doc_name === "Slow");
    releaseSlow();
    await slowResponse;
    await page.waitForTimeout(100);
    assert.equal(await input.inputValue(), "Next");
    assert.match(await page.locator(".hierarchy-section-button").innerText(), /Next section/);
    assert.doesNotMatch(await page.locator(".hierarchy-section-button").innerText(), /Slow/);
    await page.reload();
    await page.locator(".hierarchy-section-button").filter({ hasText: "Next section" }).waitFor();
    assert.equal(await input.inputValue(), "Next");
    const backRequest = page.waitForRequest((r) => r.url().endsWith("/task-layout") && r.postDataJSON().doc_name === "Slow");
    await page.goBack();
    await backRequest;
    // Let the intercepted fresh Back-navigation request enter its deferred handler.
    await page.waitForTimeout(50);
    releaseSlow();
    await page.locator(".hierarchy-section-button").filter({ hasText: "Slow section" }).waitFor();
    await input.click();
    await input.fill("Next");
    await page.getByRole("option", { name: "Next", exact: true }).click();
    await page.locator(".hierarchy-section-button").filter({ hasText: "Next section" }).waitFor();
    await input.hover();
    await page.locator(".MuiAutocomplete-clearIndicator").click();
    await input.press("Escape");
    const clearedResponse = page.waitForResponse((r) => r.url().endsWith("/Next/task-units/content"));
    await page.locator(".hierarchy-section-button").click();
    await clearedResponse;
    assert.equal(contentRequests.at(-1).path, "/api/documents/Next/task-units/content");
    assert.deepEqual(errors, []);
    assert.deepEqual(unexpected, []);
    console.log(`PASS Library identity (${staleOptions ? "stale" : "empty"} options), search typing/clear/select, A/B race, route restore/back`);
    await page.close();
  }
  {
    const page = await browser.newPage();
    const errors = [];
    const unexpected = [];
    const prepares = [];
    let ready = false;
    page.on("pageerror", (error) => errors.push(error.message));
    const guidance = "Task layout is not ready (missing_task_units). Prepare the document in Library or call POST /documents/prepare-task-layout with the same split options.";
    await page.route("**/api/**", async (route) => {
      const path = new URL(route.request().url()).pathname;
      if (path === "/api/documents" || path === "/api/documents/management") {
        return route.fulfill({ json: { items: [{ doc_name: "Book", title: "Book", source: "raw", source_type: "txt", source_status: "available", preparation_status: "ready" }], total: 1 } });
      }
      if (path === "/api/documents/task-layout") {
        return route.fulfill(ready ? { json: layout("Book") } : { status: 409, json: { detail: guidance } });
      }
      if (path === "/api/documents/upload") {
        return route.fulfill({ json: { doc_name: "Book", source_filename: "Book.txt", source_type: "txt",
          source_size_bytes: 4, source_status: "uploaded", preparation_status: "not_prepared" } });
      }
      if (path === "/api/documents/prepare-task-layout") {
        prepares.push(route.request().postDataJSON());
        if (prepares.length === 1) return route.fulfill({ status: 409, json: { detail: "Preparation readback failed." } });
        ready = true;
        return route.fulfill({ json: layout("Book") });
      }
      unexpected.push(path);
      return route.fulfill({ status: 500, json: { detail: "Unexpected API request" } });
    });
    await page.goto(base);
    await page.getByRole("button", { name: "Library", exact: true }).click();
    await page.locator(".file-management-row").filter({ hasText: "Book" })
      .getByRole("button", { name: "Open", exact: true }).click();
    const prepareDialog = page.getByRole("dialog", { name: "Prepare document", exact: true });
    await prepareDialog.waitFor();
    assert.equal(prepares.length, 0, "Open must not prepare on 409");
    await prepareDialog.getByRole("button", { name: "Close", exact: true }).click();
    await page.locator('input[type="file"]').setInputFiles({ name: "Book.txt", mimeType: "text/plain", buffer: Buffer.from("text") });
    await page.locator(".file-management-upload-action").click();
    await page.locator(".file-management-post-upload-message").filter({ hasText: "Preparation readback failed." }).waitFor();
    await page.locator(".file-management-prepare-uploaded-action").click();
    await page.locator(".file-management-post-upload-message").filter({ hasText: "Document prepared." }).waitFor();
    assert.equal(prepares.length, 2);
    assert.ok(prepares.every((request) => request.doc_name === "Book" && request.task_unit_split_mode === "progressive" && !request.layout_only));
    await page.locator(".file-management-open-uploaded-action").click();
    await page.locator(".hierarchy-section-button").filter({ hasText: "Book section" }).waitFor();
    await page.evaluate(async () => {
      const { TaskLayoutService } = await import("/src/services/TaskLayoutService.ts");
      const { RestClient } = await import("/src/services/RestClient.ts");
      await new TaskLayoutService(new RestClient()).prepareTaskLayout("Book", { layoutOnly: true });
    });
    assert.equal(prepares.at(-1).layout_only, true);
    assert.deepEqual(errors, []);
    assert.deepEqual(unexpected, []);
    console.log("PASS read-only 409 guidance, no implicit preparation, Library explicit prepare failure/retry/success, layout-only service");
    await page.close();
  }
} finally {
  await browser.close();
}
