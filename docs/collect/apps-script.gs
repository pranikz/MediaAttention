// Optional automatic collection for the MediaAttention study kit, stored in YOUR Google Drive.
//
// 1. Go to script.google.com, create a new project and paste this file.
// 2. Deploy > New deployment > type "Web app": Execute as "Me", Who has access "Anyone". Copy the /exec URL.
// 3. Put it in your study.json:  "submit_endpoint": "https://script.google.com/macros/s/.../exec"
//
// Each participant who clicks "Send results" adds one JSON file to a Drive folder named below.
// You own the data: treat it as personal data under your privacy obligations, and delete it when done.
const FOLDER = "MediaAttention results";

function doPost(e) {
  const result = JSON.parse(e.postData.contents);
  if (result.format !== "mediaattention-study/1") return ContentService.createTextOutput("rejected");
  const folders = DriveApp.getFoldersByName(FOLDER);
  const folder = folders.hasNext() ? folders.next() : DriveApp.createFolder(FOLDER);
  const id = String(result.participant).replace(/[^a-z0-9-]/gi, "").slice(0, 16);
  folder.createFile(`mediaattention-${id}.json`, e.postData.contents, "application/json");
  return ContentService.createTextOutput("ok");
}
