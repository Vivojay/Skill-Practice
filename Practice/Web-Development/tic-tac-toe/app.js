
// An array representing the tic-tac-toe board
let board = [-1, -1, -1, -1, -1, -1, -1, -1, -1];
let current_piece = 0; // First player's piece is "O"
let a = 0;

const size = board.length;
const n = Math.floor(Math.sqrt(size));
const sleep = ms => new Promise(r => setTimeout(r, ms));

function draw() {
	let fc = 0;

	for (let i = 0; i < size; i++) {
		piece = document.getElementById("cell" + (i + 1));
		piece.style.background = "linear-gradient(grey, grey)";

		fc = piece.firstChild;
		if (fc != null) {
			fc.style.color="lightgrey";
		}
	}
}

function oneSpotLeft() {
	let count = 0;
	let ind = 0;

	for (let i = 0; i < size; i++) {
		if (board[i] == -1) {
			if (!count) {
				count++;
				ind = i;
			} else {
				return null;
			}
		}
	}

	if (count == 0) return null;
	return ind;
}


function highlightWinner(winCode) {
	let winMap = {
		1: [0, 1, 2],
		2: [3, 4, 5],
		3: [6, 7, 8],

		4: [0, 3, 6],
		5: [1, 4, 7],
		6: [2, 5, 8],

		7: [0, 4, 8],
		8: [2, 4, 6]
	}

	let codes = winMap[winCode];
	let cells = document.getElementsByClassName("cell");
	let winCell = 0;

	for (let i = 0; i < size; i++) {
		winCell = cells[i];

		if (winCell.childElementCount != 0) {
			winCell.firstChild.style.color = "maroon";
		}

		// winCell.style.color = "red";
		winCell.style.background = "linear-gradient(#360703, #240604)";
	}

	for (let i = 0; i < n; i++) {
		winCell = cells[codes[i]];
		winCell.firstChild.style.color = "navy"
		winCell.style.color = "navy";
		winCell.style.background = "linear-gradient(to right, #9DCBBA, #FFEAEE)";
	}
}

function place_piece_on_screen(position, current_piece) {
	if (current_piece) {
		document.getElementById("cell" + (position + 1)).innerHTML = '<p class="ttt-cross">&times;</p>';
	} else {
		document.getElementById("cell" + (position + 1)).innerHTML = '<p class="ttt-nought">&#x2b58;</p>';
	}
}

async function place_piece(position) {

	// Position already occupied by a piece
	if (board[position] != -1) {
		alert("Position already occupied");
		return 1;
	}

	current_piece = 1 - current_piece; // Pieces alternate, so do players
	board[position] = current_piece;
	place_piece_on_screen(position, current_piece);

	a = checkwin(board);
	let oneSpotIndex = oneSpotLeft();

	if (oneSpotIndex === null) {
		console.log("lol");
	} else {
		// alert("OSL");
		// console.log(oneSpotIndex);
		// console.log(board[oneSpotIndex]);

		current_piece = 1 - current_piece; // Pieces alternate, so do players
		board[oneSpotIndex] = current_piece;
		place_piece_on_screen(oneSpotIndex, current_piece);
	}

	console.log(a);
	if (a !== null) {
		highlightWinner(a);

		await sleep(40);
		alert("Player " + ['O', 'X'][current_piece] + " won ");

		await sleep(40);
		let retry = confirm("New Game?");

		if (retry) { replay(); }

	}

	if (boardIsFull(board)) {
		console.log("loll");
		await draw();

		await sleep(40);
		let retry = confirm("It's a draw. Retry?");

		if (retry) { replay(); }
	}

	return 0;
}

function replay() {
	// reset all pieces in data structure representation
	board = [-1, -1, -1, -1, -1, -1, -1, -1, -1];
	let current_piece = 0; // First player's piece is "O"
	let piece = 0;
	let a = 0;

	// reset all pieces from user's view (HTML)
	for (let i = 0; i < size; i++) {
		piece = document.getElementById("cell" + (i + 1));
		piece.innerHTML = '';
		piece.style.background = "linear-gradient(30deg, #E0FF4F 0%, #00C49A 90%)";
		piece.style.color = null;
	}
}

function boardIsFull(board) {
	for (let i = 0; i < size; i++) {
		if (board[i] == -1) return false;
	}
	return true;
}

function checkwin(board) {

	// Check wins by row
	let flag = 0;
	let j = 0;

	for (let i = 0; i < n; i++) {
		j = n * i + 1;
		flag = 0;
		for (; j < n * i + n; j++) {
			// Check to see if any cell is empty or adj ones are nonidentical
			if (board[j] == -1 || board[j - 1] == -1 || board[j] != board[j - 1]) {
				flag = 1;
				break;
			}
		}
		// if (!flag) {return [i, j, board[j]];}
		if (!flag) { return i + 1; }
	}

	// console.log("col-check");
	// Check wins by cols
	flag = 0;
	j = 0;

	for (let i = 0; i < n; i++) {
		j = i + n;
		flag = 0;
		for (; j < i + 2 * n + 1; j += n) {
			// console.log("j-n: "+(j-n)+", j: "+j);
			// Check to see if any cell is empty or adj ones are unidentical
			if (board[j] == -1 || board[j - n] == -1 || board[j] != board[j - n]) {
				flag = 1;
				break;
			}
		}

		if (!flag) { return i + 4; }
	}

	// Check wins by diags
	let interval1 = (n + 1);
	let interval2 = (n - 1);
	let d1 = n + 1;
	let d2 = 2*n - 2;
	// let flag1 = 0;
	// let flag2 = 0;

	flag = 0;
	for (; d1 < size; d1 += interval1) {
		if (board[d1] == -1 || board[d1 - interval1] == -1 || board[d1] != board[d1 - interval1]) {
			flag = 1;
			break;
		}
	}
	
	if (!flag) {return 7;}
	
	flag = 0;
	for (; d2 < size-n+1; d2 += interval2) {
		if (board[d2] == -1 || board[d2 - interval2] == -1 || board[d2] != board[d2 - interval2]) {
			flag = 1;
			break;
		}
	}
	
	if (!flag) {return 8;}

	/*
	// Check wins by diags
	let interval1 = (n + 1);
	let interval2 = (n - 1);
	let d1 = n + 1;
	let d2 = 2*n - 2;
	let flag1 = 0;
	let flag2 = 0;

	while (d1 < size) {
		flag1 = 0;
		flag2 = 0;

		// Check to see if any cell in diag1 is empty or ones adj to it along diag1 are unidentical
		if (flag1 == 0) {
			if (board[d1] == -1 || board[d1 - interval1] == -1 || board[d1] != board[d1 - interval1]) {
				flag1 = 1;
			}
		}

		// Check to see if any cell in diag2 is empty or ones adj to it along diag1 are unidentical
		if (flag2 == 0) {
			if (board[d2] == -1 || board[d2 - interval2] == -1 || board[d2] != board[d2 - interval2]) {
				flag2 = 1;
			}
		}
		d1 += interval1;
		d2 += interval2;
	}


	if (!flag1) { return 7; }
	if (!flag2) { return 8; }
	*/

	return null;
}


