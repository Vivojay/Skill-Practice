function setRandomDiceFaces() {
    var diceNumbersArray = [];

    var randomDiceNumber = Math.floor(Math.random() * 6) + 1;
    var diceImagePath1 = "./images/dice" + randomDiceNumber + ".png";
    diceNumbersArray.push(randomDiceNumber);
    
    randomDiceNumber = Math.floor(Math.random() * 6) + 1;
    var diceImagePath2 = "./images/dice" + randomDiceNumber + ".png";
    diceNumbersArray.push(randomDiceNumber);
    
    document.querySelector(".dice-img-1").src = diceImagePath1;
    document.querySelector(".dice-img-2").src = diceImagePath2;

    return diceNumbersArray;
}

function setStatus(diceNumbersArray) {
    var statusHTML = "initStatus";

    if (diceNumbersArray[0] > diceNumbersArray[1]) {
        statusHTML = "🚩 Player 1 Wins!";
    }
    else if (diceNumbersArray[0] < diceNumbersArray[1]) {
        statusHTML = "Player 2 Wins! 🚩";
    }
    else {
        statusHTML = "<i class='fa-solid fa-equals'></i> Draw! <i class='fa-solid fa-equals'></i>";
    }
    document.querySelector(".main-heading").innerHTML = statusHTML;
}

setStatus(setRandomDiceFaces());
