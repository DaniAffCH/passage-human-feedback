const csvHeader = "ip_sender,r1t1_x,r1t1_y,r2t1_x,r2t1_y,r3t1_x,r3t1_y,r4t1_x,r4t1_y,r5t1_x,r5t1_y,r6t1_x,r6t1_y,r1t2_x,r1t2_y,r2t2_x,r2t2_y,r3t2_x,r3t2_y,r4t2_x,r4t2_y,r5t2_x,r5t2_y,r6t2_x,r6t2_y,ball_holder,pass_to"
const fileName = "output.csv"

const express = require('express');
const fs = require('fs');
const bodyParser = require('body-parser');

const app = express();
const port = 3000;
app.use(express.urlencoded({ extended: true }));
app.use(express.static('public', { 'extensions': ['css'] }));
app.use(express.static('assets'));
app.use(bodyParser.json());

app.use((req, res, next) => {
    const { 'user-robocup': userRobocup, 'user-soccer': userSoccer } = req.query;
    const allowedRc = ["participated", "attended", "no"] 
    const allowedSo = ["yes", "no"]

    if (req.path === '/game' && (!allowedRc.includes(userRobocup) || !allowedSo.includes(userSoccer))) {
        return res.redirect('/');
    }

    next();
});

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html');

});

app.get('/game', (req, res) => {
  res.sendFile(__dirname + '/game.html');

});

app.post('/savecsv', (req, res) => {
    const { csvContent } = req.body;
    const clientIp = req.headers['x-forwarded-for'] || req.socket.remoteAddress;

    if (!csvContent) {
        return res.status(400).json({ error: 'CSV content is required' });
    }

    // Check if the file exists
    fs.access(fileName, fs.constants.F_OK, (err) => {
        if (err) {
            // If the file doesn't exist, create a new one
            fs.writeFile(fileName, csvHeader+"\n"+clientIp+","+csvContent, (writeErr) => {
                if (writeErr) {
                    console.error(writeErr);
                    return res.status(500).json({ error: 'Error saving CSV file' });
                }

                res.json({ success: true });
            });
        } else {
            // If the file exists, append the new content
            fs.appendFile(fileName, '\n' +clientIp+","+ csvContent, (appendErr) => {
                if (appendErr) {
                    console.error(appendErr);
                    return res.status(500).json({ error: 'Error appending to CSV file' });
                }

                res.json({ success: true });
            });
        }
    });
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
