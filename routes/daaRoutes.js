const express = require('express');
const router = express.Router();
const daaResponses = require('../data/daaResponses');

// DAA routes handler
router.get('/:id', (req, res) => {
  const { id } = req.params;
  const response = daaResponses[id];
  
  if (response) {
    res.send(response);
  } else {
    res.status(404).send('DAA route not found');
  }
});

module.exports = router;