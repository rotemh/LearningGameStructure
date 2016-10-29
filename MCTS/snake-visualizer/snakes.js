var state = {};
var canvas = document.getElementById('game');
var cellSize = 28;
canvas.width = cellSize * 40;
canvas.height = cellSize * 25;
var gfx = canvas.getContext('2d');
var worker = new Worker('snorker.js');
worker.onmessage = function(message) {
  state = message.data;
};


function cx(cell) {
  return cell[0] * cellSize;
}
function cy(cell) {
  return (24 - cell[1]) * cellSize;
}

function drawFood(food) {
  gfx.fillStyle = '#68CC00';
  for (var i = 0; i < food.length; i++) {
    var x = cx(food[i]);
    var y = cy(food[i]);
    gfx.fillRect(x, y, cellSize, cellSize);
  }
}


function drawSnake(snake, dir, color) {
  gfx.fillStyle = color;
  for (var i = 1; i < snake.length - 1; i++) {
    var x = cx(snake[i]);
    var y = cy(snake[i]);
    gfx.fillRect(x, y, cellSize, cellSize);
  }
  drawSnakeHead(snake, color);
  drawSnakeTail(snake, color);
}

function getDirAngle(dir) {
  // Left
  if (dir === 1) {
    return Math.PI;
  }
  // Up
  if (dir === 2) {
    return 3 * Math.PI / 2;
  }
  // Right
  if (dir === 3) {
    return 0;
  }
  // Down
  if (dir === 4) {
    return Math.PI / 2;
  }
  throw new Error('Unrecognized dir: ' + dir);
}


function drawSnakeCap(x, y, dir, color) {
  var capX = x;
  var capY = y;
  var centerX = x + cellSize / 2;
  var centerY = y + cellSize / 2;
  var capWidth = cellSize;
  var capHeight = cellSize;
  var dirAngle = getDirAngle(dir);
  if (dir === 1 || dir === 3) {
    capWidth /= 2;
  }
  if (dir === 2 || dir === 4) {
    capHeight /= 2;
  }
  if (dir === 1) {
    capX = centerX;
  }
  if (dir === 2) {
    capY = centerY;
  }

  var startAngle = dirAngle - Math.PI / 2;
  var endAngle = dirAngle + Math.PI / 2;

  gfx.fillStyle = color;
  gfx.beginPath()
  gfx.arc(centerX, centerY, cellSize / 2, startAngle, endAngle);
  gfx.closePath();
  gfx.fill();
  gfx.fillRect(capX, capY, capWidth, capHeight);
}

function drawSnakeHead(snake, color) {
  var x = cx(snake[0]);
  var y = cy(snake[0]);
  var centerX = x + cellSize / 2;
  var centerY = y + cellSize / 2;
  var dir = getDir(snake[0], snake[1]);
  var dirAngle = getDirAngle(dir);
  drawSnakeCap(x, y, dir, color);

  gfx.save();
  gfx.fillStyle = 'white';
  gfx.translate(centerX, centerY);
  gfx.rotate(dirAngle);

  gfx.beginPath();
  gfx.arc(cellSize * 0.2, -cellSize * 0.2, cellSize * 0.17, 0, 2 * Math.PI);
  gfx.fill();
  gfx.beginPath();
  gfx.arc(cellSize * 0.2, cellSize * 0.2, cellSize * 0.17, 0, 2 * Math.PI);
  gfx.fill();

  gfx.fillStyle = 'black';
  gfx.beginPath();
  gfx.arc(0.23 * cellSize, -0.2 * cellSize, cellSize * 0.1, 0, 2 * Math.PI);
  gfx.fill();
  gfx.beginPath();
  gfx.arc(0.23 * cellSize, 0.2 * cellSize, cellSize * 0.1, 0, 2 * Math.PI);
  gfx.fill();
  gfx.restore();
}

function getDir(cellA, cellB) {
  var tailX = cx(cellA);
  var tailY = cy(cellA);
  var preTailX = cx(cellB);
  var preTailY = cy(cellB);

  var diffX = tailX - preTailX;
  var diffY = tailY - preTailY;
  if (Math.abs(diffX) > Math.abs(diffY)) {
    if (diffX > 0) {
      return 3;
    } else {
      return 1;
    }
  } else {
    if (diffY > 0) {
      return 4;
    } else {
      return 2;
    }
  }
}

function drawSnakeTail(snake, color) {
  var tailX = cx(snake[snake.length - 1]);
  var tailY = cy(snake[snake.length - 1]);
  var dir = getDir(snake[snake.length - 1], snake[snake.length - 2]);
  drawSnakeCap(tailX, tailY, dir, color);
}


function draw() {
  gfx.fillStyle = 'rgba(0, 0, 0, 0.15)';
  gfx.fillRect(0, 0, canvas.width, canvas.height);

  if (state.gameId) {
    var redSnake = state.redSnake;
    var blackSnake = state.blackSnake;

    // for (var row = 0; row < map.length; row++) {
    // }
    // for (var col = 0; col < map[row].length; col++) {
    // }
    drawFood(state.food);
    drawSnake(state.redSnake, state.redDir, '#FF3900');
    drawSnake(state.blackSnake, state.blackDir, '#0035FF');
  }


  window.requestAnimationFrame(draw);
}

draw();
