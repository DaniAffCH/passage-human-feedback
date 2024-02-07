const csvHeader = "ip_sender,timestamp,roboCup_familiarity, soccer_familiarity, age, gender ,r1t1_x,r1t1_y,r2t1_x,r2t1_y,r3t1_x,r3t1_y,r4t1_x,r4t1_y,r5t1_x,r5t1_y,r6t1_x,r6t1_y,r7t1_x,r7t1_y,r1t2_x,r1t2_y,r2t2_x,r2t2_y,r3t2_x,r3t2_y,r4t2_x,r4t2_y,r5t2_x,r5t2_y,r6t2_x,r6t2_y,r7t2_x,r7t2_y,ball_holder,pass_to"
const fileName = "data_v0.csv"

const express = require('express');
const fs = require('fs');
const session = require('express-session');
const bodyParser = require('body-parser');

const app = express();
const port = 3000;
app.use(express.urlencoded({ extended: true }));
app.use(express.static('public', { 'extensions': ['css'] }));
app.use(express.static('assets'));
app.use(bodyParser.json());

app.use(session({
    secret: 'robocupspqr', 
    resave: false,
    saveUninitialized: true
}));


app.use((req, res, next) => {
    const { 'user-robocup': userRobocup, 'user-soccer': userSoccer, 'user-age': userAge, 'user-gender': userGender } = req.query;
    const allowedRc = ["1", "2", "3", "4", "5"] 
    const allowedSo = ["1", "2", "3", "4", "5"]
    const allowedUa = ["<18", "18-22", "23-30", "31-40", "41-50", ">50"]
    const allowedUg = ["m","f","n"]

    if (req.path === '/game' && (!allowedRc.includes(userRobocup) || !allowedSo.includes(userSoccer) || !allowedUa.includes(userAge) || !allowedUg.includes(userGender))) {
        return res.redirect('/');
    }

    if (userRobocup && userSoccer && userAge && userGender) {
        req.session.userRobocup = userRobocup;
        req.session.userSoccer = userSoccer;
        req.session.userAge = userAge;
        req.session.userGender = userGender;
    }    

    next();
});

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html');

});

app.get('/game', (req, res) => {
  res.sendFile(__dirname + '/game.html');

});

function getCurrentDateTime() {
  const now = new Date();

  const day = String(now.getDate()).padStart(2, '0');
  const month = String(now.getMonth() + 1).padStart(2, '0'); // Months are zero-based
  const year = now.getFullYear();

  const hours = String(now.getHours()).padStart(2, '0');
  const minutes = String(now.getMinutes()).padStart(2, '0');
  const seconds = String(now.getSeconds()).padStart(2, '0');

  const formattedDate = `${day}/${month}/${year} ${hours}:${minutes}:${seconds}`;
  return formattedDate;
}

app.post('/savecsv', (req, res) => {
    const { csvContent } = req.body;
    const clientIp = req.headers['x-forwarded-for'] || req.socket.remoteAddress;

    if (!csvContent) {
        return res.status(400).json({ error: 'CSV content is required' });
    }
    const dt = getCurrentDateTime()
    const row = `${clientIp},${dt},${req.session.userRobocup},${req.session.userSoccer},${req.session.userAge},${req.session.userGender},${csvContent}`

    // Check if the file exists
    fs.access(fileName, fs.constants.F_OK, (err) => {
        if (err) {
            // If the file doesn't exist, create a new one
            fs.writeFile(fileName, csvHeader+"\n"+row, (writeErr) => {
                if (writeErr) {
                    console.error(writeErr);
                    return res.status(500).json({ error: 'Error saving CSV file' });
                }
                
                res.json({ success: true });
            });
        } else {
            // If the file exists, append the new content
            fs.appendFile(fileName, '\n'+row, (appendErr) => {
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
