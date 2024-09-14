// static/form-validation.js

function validateForm() {
  // Get the form element
  var form = document.getElementById('prediction-form');
  
  // Get all input and select elements that are required
  var inputs = form.querySelectorAll('input[required], select[required]');
  
  // Check if all required fields are filled
  var allFilled = Array.from(inputs).every(input => input.value.trim() !== '');
  
  if (!allFilled) {
      // Show an alert if some fields are not filled
      alert('Some input columns are not filled');
      return false; // Prevent form submission
  }
  return true; // Allow form submission
}
