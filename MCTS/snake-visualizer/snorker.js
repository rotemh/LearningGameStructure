var lastResponse = '';

function pullState() {
  var xhr = new XMLHttpRequest();
  xhr.open('GET', '../vis/game-state.json', true);
  xhr.overrideMimeType('text/plain');
  xhr.onload = function() {
    setTimeout(pullState, 50);
    // Hacky deduplication
    if (lastResponse === xhr.responseText) {
      return;
    }
    var newState = JSON.parse(xhr.responseText);
    self.postMessage(newState);
  };
  xhr.onerror = function() {
    setTimeout(pullState, 50);
  };
  xhr.send();
}

pullState();
