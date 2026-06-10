import fs from 'fs';
import path from 'path';

const mapRegex = [
  { rx: /bg-\[\#000000\]/g, rep: 'bg-background' },
  { rx: /bg-\[\#0A0A0A\]/g, rep: 'bg-card' },
  { rx: /bg-\[\#171717\]/g, rep: 'bg-secondary' },
  { rx: /bg-\[\#27272A\]/g, rep: 'bg-input' },
  { rx: /bg-\[\#EDEDED\]/g, rep: 'bg-foreground' },
  { rx: /border-\[\#27272A\]/g, rep: 'border-border' },
  { rx: /border-\[\#52525B\]/g, rep: 'border-muted-foreground' },
  { rx: /divide-\[\#27272A\]/g, rep: 'divide-border' },
  { rx: /text-\[\#EDEDED\]/g, rep: 'text-foreground' },
  { rx: /text-\[\#A1A1AA\]/g, rep: 'text-muted-foreground' },
  { rx: /text-\[\#52525B\]/g, rep: 'text-muted-foreground' },
  { rx: /text-\[\#000000\]/g, rep: 'text-background' },
  { rx: /placeholder:text-\[\#52525B\]/g, rep: 'placeholder:text-muted-foreground' },
  { rx: /placeholder-\[\#52525B\]/g, rep: 'placeholder-muted-foreground' },
  { rx: /fill-\[\#EDEDED\]/g, rep: 'fill-foreground' },
  { rx: /fill-\[\#A1A1AA\]/g, rep: 'fill-muted-foreground' },
  { rx: /stroke-\[\#EDEDED\]/g, rep: 'stroke-foreground' },
  { rx: /stroke-\[\#A1A1AA\]/g, rep: 'stroke-muted-foreground' },
  { rx: /ring-\[\#EDEDED\]/g, rep: 'ring-ring' },
];

function processDirectory(dirPath) {
  const entries = fs.readdirSync(dirPath, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dirPath, entry.name);
    
    if (entry.isDirectory()) {
      processDirectory(fullPath);
    } else if (entry.isFile() && (fullPath.endsWith('.tsx') || fullPath.endsWith('.ts'))) {
      let content = fs.readFileSync(fullPath, 'utf8');
      let modified = false;

      for (const { rx, rep } of mapRegex) {
        if (rx.test(content)) {
          content = content.replace(rx, rep);
          modified = true;
        }
      }

      if (modified) {
        fs.writeFileSync(fullPath, content, 'utf8');
        console.log(`Updated: ${fullPath}`);
      }
    }
  }
}

const targetDir = 'c:\\Users\\Prashanth yadav\\OneDrive - jbrec.edu.in\\Documents\\Coding\\CipherVerse\\frontend\\src';
processDirectory(targetDir);
console.log('Done!');
