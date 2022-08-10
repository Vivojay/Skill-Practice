var drumButtons = document.querySelectorAll(".drum");

// Add button press event listener to all drum sounds
for (var i = 0; i < drumButtons.length; i++) {
    drumButtons[i].addEventListener("click", function () {
        playSound(this.innerHTML);
        buttonAnimation(this.innerHTML);
    });
}

// Add keyboard key-down event listener to all drum sounds
document.addEventListener("keydown", function (event) { playSound(event.key) });

function playSound(buttonLetter) {
    switch (buttonLetter) {
        case 'w':
            var soundName = "wooden-clave"; break;
        case 'a':
            var soundName = "hi-fx"; break;
        case 's':
            var soundName = "glass-break-fx"; break;
        case 'd':
            var soundName = "o-hi-hat"; break;
        case 'j':
            var soundName = "snare"; break;
        case 'k':
            var soundName = "c-hi-hat"; break;
        case 'l':
            var soundName = "kick-drum"; break;

        default:
            console.log(buttonLetter);
    }
    playButtonSound(soundName);
    buttonAnimation(buttonLetter);
}

function playButtonSound(soundName) {
    var audio = new Audio("sounds/" + soundName + ".mp3");
    audio.play();
}

function buttonAnimation(currentKey) {
    document.querySelector("." + currentKey).classList.add("pressed");
    myTimeout = setTimeout(function () {
        document.querySelector("." + currentKey).classList.remove("pressed");
    }, 100);
}
