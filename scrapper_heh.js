const { Builder, By, until } = require("selenium-webdriver");
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

(async function scrape() {
  let driver = await new Builder().forBrowser("chrome").build();
  const url =
    "https://crustdata.notion.site/Crustdata-Discovery-And-Enrichment-API-c66d5236e8ea40df8af114f6d447ab48";

  try {
    await driver.get(url);
    await driver.wait(
      until.elementLocated(By.css(".notion-page-content")),
      20000
    );

    // Click buttons only within .notion-page-content
    let container = await driver.findElement(By.css(".notion-page-content"));
    let buttons = await container.findElements(By.css('div[role="button"]'));
    for (let button of buttons) {
      try {
        await driver.executeScript("arguments[0].click();", button);
        await driver.sleep(1000); // Wait a bit more to ensure content loads
      } catch (e) {
        console.log(`Could not click button: ${e}`);
      }
    }

    // Get the updated page source
    let pageSource = await driver.getPageSource();

    // Parse the HTML with JSDOM
    const dom = new JSDOM(pageSource);
    const document = dom.window.document;
    const content = document.querySelector(".notion-page-content");

    // Extract and structure all text content
    let sections = {};
    let currentTitle = "untitled_section";

    content.querySelectorAll("*").forEach((element) => {
      let text = element.textContent.trim();

      if (text) {
        let tagName = element.tagName.toLowerCase();

        // Start new section when encountering an h3 tag
        if (tagName === "h3") {
          currentTitle = text.replace(/[^a-zA-Z0-9]/g, "_"); // Clean title for file name
          if (!sections[currentTitle]) {
            sections[currentTitle] = [];
          }
        }

        // Ensure the section exists before checking for duplicates
        if (sections[currentTitle] && !sections[currentTitle].includes(text)) {
          sections[currentTitle].push(text);
        }
      }
    });

    // Save each section to a separate file
    const outputDir = "scraped_data_new_gpt";
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir);
    }

    for (const [title, contentArray] of Object.entries(sections)) {
      let sectionContent = contentArray.join("\n");
      fs.writeFileSync(
        path.join(outputDir, `${title}.txt`),
        sectionContent,
        "utf-8"
      );
    }

    console.log("Scraped and saved sections as separate files.");
  } finally {
    await driver.quit();
  }
})();
