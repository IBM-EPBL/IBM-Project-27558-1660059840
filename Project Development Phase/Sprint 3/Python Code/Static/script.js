var btnLogin = document.getElementById('do-login');
var btnRun = document.getElementById('do-run');
var idLogin = document.getElementById('login');
var username = document.getElementById('username');
btnLogin.onclick = function(){
  idLogin.innerHTML = '<p>Hi, </p><h1>' +username.value+ '</h1><p>We hope you are safe :)<br>This application aims to solve real life problem of disaster identification <br>along with its estimated intensity on a level of 1-5.<br>1 denotes low level intensity while 5 denotes the highest intensity level.The users are requested<br> to use the run button to upload good, clarity pictures to run our convolutional network ... </p><a href="run.html" style="font-size: 24px;">Run</a>';
}
btnRun.onclick = function(){
  idLogin.innerHTML = '<p>Hi, </p><h1>' +username.value+ '</h1><form action="/action_page.php"><input type="file" id="myFile" name="filename"><br><input type="Submit"></form>';
}