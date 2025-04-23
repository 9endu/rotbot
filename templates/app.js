// Initialize Firebase
const firebaseConfig = {
    apiKey: "AIzaSyAOk65Mg_P1ISLbGRB6I-3HDL-U-hCFO-c",
    authDomain: "rotbot-b300b.firebaseapp.com",
    databaseURL: "https://rotbot-b300b-default-rtdb.firebaseio.com",
    projectId: "rotbot-b300b",
    storageBucket: "rotbot-b300b.appspot.com",
    messagingSenderId: "1022382810982",
    appId: "1:1022382810982:web:cb0f1a631f857de4a18aa7",
    measurementId: "G-B6S4DC2L03"
}

firebase.initializeApp(firebaseConfig);
const database = firebase.database();

// Store the email for use in step 2
let userEmail = '';

function sendResetLink() {
    const email = document.getElementById('email').value.trim();
    userEmail = email;
    
    if (!email) {
        showMessage('Please enter your email address', 'error');
        return;
    }
    
    // Check if email exists in the database
    database.ref('companies').orderByChild('email').equalTo(email).once('value')
        .then(snapshot => {
            if (snapshot.exists()) {
                // Generate a simple verification code (in production, use a more secure method)
                const verificationCode = Math.floor(100000 + Math.random() * 900000).toString();
                
                // Store the code temporarily in the database (or use Firebase Auth's password reset)
                database.ref('passwordResetCodes').child(email.replace('.', ',')).set({
                    code: verificationCode,
                    timestamp: Date.now()
                });
                
                // In a real app, you would send this code via email
                console.log(`Verification code for ${email}: ${verificationCode}`);
                
                // Show step 2
                document.getElementById('step1').style.display = 'none';
                document.getElementById('step2').style.display = 'block';
                showMessage(`A verification code has been sent to ${email} (check console for demo)`, 'success');
            } else {
                showMessage('No account found with that email address', 'error');
            }
        })
        .catch(error => {
            showMessage('Error: ' + error.message, 'error');
        });
}

function resetPassword() {
    const code = document.getElementById('code').value.trim();
    const newPassword = document.getElementById('newPassword').value;
    
    if (!code || !newPassword) {
        showMessage('Please enter both the verification code and new password', 'error');
        return;
    }
    
    if (newPassword.length < 6) {
        showMessage('Password should be at least 6 characters', 'error');
        return;
    }
    
    // Verify the code
    database.ref('passwordResetCodes').child(userEmail.replace('.', ',')).once('value')
        .then(snapshot => {
            const data = snapshot.val();
            
            // Check if code exists and is not expired (5 minutes)
            if (data && data.code === code && Date.now() - data.timestamp < 300000) {
                // Code is valid - update password in the database
                return database.ref('companies').orderByChild('email').equalTo(userEmail).once('value');
            } else {
                throw new Error('Invalid or expired verification code');
            }
        })
        .then(snapshot => {
            if (snapshot.exists()) {
                // Find the company with this email (there should be only one)
                let companyKey = null;
                let companyData = null;
                
                snapshot.forEach(childSnapshot => {
                    companyKey = childSnapshot.key;
                    companyData = childSnapshot.val();
                });
                
                if (companyKey) {
                    // Update the password
                    return database.ref('companies/' + companyKey).update({
                        password: newPassword // In production, hash this password first!
                    });
                } else {
                    throw new Error('Company not found');
                }
            } else {
                throw new Error('Company not found');
            }
        })
        .then(() => {
            // Clean up the verification code
            database.ref('passwordResetCodes').child(userEmail.replace('.', ',')).remove();
            
            showMessage('Password updated successfully! You can now login with your new password.', 'success');
            setTimeout(() => {
                window.location.href = 'login.html'; // Redirect to login page
            }, 3000);
        })
        .catch(error => {
            showMessage('Error: ' + error.message, 'error');
        });
}

function showMessage(message, type) {
    const messageDiv = document.getElementById('message');
    messageDiv.textContent = message;
    messageDiv.className = 'message ' + type;
    messageDiv.style.display = 'block';
}