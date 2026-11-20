const path = require('path');
const { pathToFileURL } = require('url');
const { chromium } = require('playwright');

(async () => {
  const guideDir = path.resolve(__dirname, '..');
  const workDir = path.join(guideDir, 'working');
  const outputPdf = path.join(guideDir, 'OpenCodeReview_AI_Engineering_Architecture_Guide.pdf');
  const launchOptions = { headless: true };
  if (process.env.BROWSER_EXECUTABLE_PATH) {
    launchOptions.executablePath = process.env.BROWSER_EXECUTABLE_PATH;
  }
  const browser = await chromium.launch(launchOptions);
  const page = await browser.newPage({ viewport: { width: 1275, height: 1650 } });
  await page.goto(pathToFileURL(path.join(workDir, 'preview.html')).href, { waitUntil: 'load' });
  await page.pdf({
    path: outputPdf,
    format: 'Letter',
    printBackground: true,
    preferCSSPageSize: true,
    displayHeaderFooter: true,
    headerTemplate: '<div style="font-family:Arial,sans-serif;font-size:8px;color:#52606D;width:100%;text-align:right;padding-right:0.78in;">OpenCodeReview AI Engineering Guide</div>',
    footerTemplate: '<div style="font-family:Arial,sans-serif;font-size:8px;color:#52606D;width:100%;text-align:center;">OpenCodeReview | Repository snapshot d939059 | Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>',
  });
  await browser.close();
  console.log(outputPdf);
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
