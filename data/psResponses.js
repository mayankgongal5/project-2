const psResponses = {
  1: `<!--Here is an HTML document demonstrating character formatting (10 examples) and page formatting (5 examples), along with creating divisions, inserting an image, and explaining the code:-->

  <!--  HTML Code-->

    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HTML Formatting Examples</title>
        <style>
            /* Page formatting examples */
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            .center { text-align: center; }
            .container { padding: 20px; border: 1px solid #ccc; background-color: #f9f9f9; }
            .highlight { background-color: yellow; }
            .image-container { text-align: center; }
        </style>
    </head>
    <body>
        <!-- Page formatting: Title, margins, and alignment -->
        <h1 class="center">HTML Formatting Examples</h1>

        <!-- Division for content -->
        <div class="container">
            <!-- Character formatting examples -->
            <p><b>1. Bold text</b> using the <code>&lt;b&gt;</code> tag.</p>
            <p><i>2. Italicized text</i> with the <code>&lt;i&gt;</code> tag.</p>
            <p><u>3. Underlined text</u> using <code>&lt;u&gt;</code>.</p>
            <p>4. <mark>Highlighted text</mark> using the <code>&lt;mark&gt;</code> tag.</p>
            <p>5. <small>Small text</small> for captions or fine print.</p>
            <p>6. <strong>Strong text</strong> emphasizes importance.</p>
            <p>7. <em>Emphasized text</em> using <code>&lt;em&gt;</code>.</p>
            <p>8. <sub>Subscript</sub> and <sup>Superscript</sup> for scientific expressions like H<sub>2</sub>O or x<sup>2</sup>.</p>
            <p>9. Text with a <del>strikethrough</del> effect.</p>
            <p>10. Monospaced font for <code>inline code</code>.</p>
        </div>

        <!-- Page formatting: Image example -->
        <div class="image-container">
            <h2>Insert an Image</h2>
            <img src="example-image.jpg" alt="Sample Image" width="300" height="200">
        </div>
    </body>
    </html>

   <!-- Explanation-->
	<!--•	Character Formatting: Tags like <b>, <i>, <u>, <strong>, <small>, <sub>, <sup>, <mark>, <del>, and <code> demonstrate styling individual text content.-->
	<!--•	Page Formatting: Used body margins, padding, text alignment (center), background colors, and a styled container to organize content.-->
	<!--•	Divisions: <div> creates logical sections with a class-based styling.-->
	<!--•	Image: The <img> tag inserts an image, providing attributes like src (path), alt (description), and dimensions.-->

   <!-- This combines formatting and organization with a clean structure.-->`,

  2: `<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Student Mark Sheet</title>
        <style>
            /* Table styling */
            table {
                width: 80%;
                border-collapse: collapse;
                margin: 20px auto;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
            }
            th, td {
                border: 1px solid #333;
                text-align: center;
                padding: 10px;
            }
            th {
                background-color: #4CAF50;
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            tr:hover {
                background-color: #ddd;
            }
            caption {
                font-size: 1.5em;
                margin: 10px;
                font-weight: bold;
                color: #333;
            }
            .total {
                font-weight: bold;
                background-color: #ffd700;
            }
        </style>
    </head>
    <body>
        <!-- Table example: Mark Sheet -->
        <table>
            <caption>Student Mark Sheet</caption>
            <thead>
                <tr>
                    <th>Roll No</th>
                    <th>Student Name</th>
                    <th>Subject</th>
                    <th>Marks Obtained</th>
                    <th>Max Marks</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>101</td>
                    <td>Mayank</td>
                    <td>Mathematics</td>
                    <td>85</td>
                    <td>100</td>
                    <td>85%</td>
                </tr>
                <tr>
                    <td>102</td>
                    <td>Shubhankar</td>
                    <td>Physics</td>
                    <td>78</td>
                    <td>100</td>
                    <td>78%</td>
                </tr>
                <tr>
                    <td>103</td>
                    <td>Ketan</td>
                    <td>PS</td>
                    <td>92</td>
                    <td>100</td>
                    <td>92%</td>
                </tr>
                <tr>
                    <td>104</td>
                    <td>Gaurav</td>
                    <td>DAA</td>
                    <td>88</td>
                    <td>100</td>
                    <td>88%</td>
                </tr>
                <tr>
                    <td>105</td>
                    <td>Bassi</td>
                    <td>AI</td>
                    <td>74</td>
                    <td>100</td>
                    <td>74%</td>
                </tr>
            </tbody>
            <tfoot>
                <tr class="total">
                    <td colspan="3">Total</td>
                    <td>417</td>
                    <td>500</td>
                    <td>83.4%</td>
                </tr>
            </tfoot>
        </table>
    </body>
    </html>

   <!-- Explanation:-->
	<!--1.	HTML Table Attributes:-->
	<!--•	Used attributes like <th>, <td>, <tr>, <caption>, <thead>, <tbody>, and <tfoot>.-->
	<!--•	Used colspan for merging cells in the Total row.-->
	<!--2.	CSS Table Formatting:-->
	<!--•	border-collapse: Ensures single borders between cells.-->
	<!--•	box-shadow: Adds a shadow for better aesthetics.-->
	<!--•	nth-child and hover: For row striping and hover effects.-->
	<!--•	caption: Adds a table heading.-->
	<!--3.	Logical Structure:-->
	<!--•	The table represents a Mark Sheet with sections for roll number, names, marks, and total percentage.-->
	<!--4.	Total Row:-->
	<!--•	Added bold styling and a distinct background color for the total row.-->`,

  3: `<!--Here is a simple registration form using HTML form objects with minimal code:-->

  <!--  HTML Code-->

    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Registration Form</title>
    </head>
    <body>
        <h2>Registration Form</h2>
        <form action="#" method="post">
            <label>First Name:</label>
            <input type="text" name="first_name" required><br><br>

            <label>Last Name:</label>
            <input type="text" name="last_name" required><br><br>

            <label>Email:</label>
            <input type="email" name="email" required><br><br>

            <label>Password:</label>
            <input type="password" name="password" required><br><br>

            <label>Gender:</label>
            <input type="radio" name="gender" value="male"> Male
            <input type="radio" name="gender" value="female"> Female <br><br>

            <label>Country:</label>
            <select name="country">
                <option value="India">India</option>
                <option value="USA">USA</option>
                <option value="UK">UK</option>
            </select><br><br>

            <label>Comments:</label><br>
            <textarea name="comments" rows="4" cols="40"></textarea><br><br>

            <input type="submit" value="Register">
            <input type="reset" value="Reset">
        </form>
    </body>
    </html>

   <!-- Key Features:-->
	<!--1.	Form Objects Used:-->
	<!--•	input (text, email, password, radio, submit, reset)-->
	<!--•	textarea for multiline input-->
	<!--•	select dropdown for country selection.-->
	<!--2.	Minimal Code:-->
	<!--•	No external or internal CSS is used.-->
	<!--•	Basic <label> and form fields provide clear structure.-->
	<!--3.	Accessibility:-->
	<!--•	required ensures mandatory fields like First Name, Last Name, Email, and Password.-->

   <!-- This form collects essential user details like name, email, password, gender, and comments.-->`,

  4: `<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Website Layout</title>
    </head>
    <body style="margin: 0; font-family: Arial, sans-serif;">

        <table border="1" width="100%" cellspacing="0" cellpadding="10" style="border-collapse: collapse;">

            <tr style="background-color: #4CAF50; color: white;">
                <th colspan="2" style="text-align: center; font-size: 24px;">My Simple Website</th>
            </tr>


            <tr style="background-color: #f2f2f2; color: #333;">
                <td colspan="2" style="text-align: center;">
                    <a href="#" style="text-decoration: none; color: #4CAF50; margin: 0 10px;">Home</a>
                    <a href="#" style="text-decoration: none; color: #4CAF50; margin: 0 10px;">About</a>
                    <a href="#" style="text-decoration: none; color: #4CAF50; margin: 0 10px;">Services</a>
                    <a href="#" style="text-decoration: none; color: #4CAF50; margin: 0 10px;">Contact</a>
                </td>
            </tr>


            <tr>

                <td width="20%" style="background-color: #e7e7e7; color: #333;">
                    <h3>Sidebar</h3>
                    <p>Quick Links:</p>
                    <ul>
                        <li>Link 1</li>
                        <li>Link 2</li>
                        <li>Link 3</li>
                    </ul>
                </td>


                <td style="background-color: #fff; color: #333;">
                    <h2>Welcome to My Website</h2>
                    <p>This my portfolio website.</p>
                    <p>I will put my resume here.</p>
                </td>
            </tr>


            <tr style="background-color: #4CAF50; color: white;">
                <td colspan="2" style="text-align: center;">&copy; 2024 My Website | All Rights Reserved</td>
            </tr>
        </table>

    </body>
    </html>
   <!-- Features:-->
	<!--1.	Table Layout:-->
	<!--•	The website layout is implemented using a table with header, navigation, sidebar, content, and footer.-->
	<!--2.	CSS:-->
	<!--•	Inline CSS is used for minimal code:-->
	<!--•	background-color for different sections.-->
	<!--•	color for text.-->
	<!--•	Border styling and alignment.-->
	<!--3.	Responsive Table:-->
	<!--•	Table width is set to 100% to ensure it adjusts to any screen size.-->`,

  5: `<!--Here is the JavaScript code with minimal implementation to display the text “TEXT-GROWING” and “TEXT-SHRINKING” with the described behavior:-->

  <!--  HTML + JavaScript Code-->

    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Text Growing and Shrinking</title>
    </head>
    <body>
        <div id="text" style="font-size: 5pt; color: red;">TEXT-GROWING</div>

        <script>
            let size = 5;
            let growing = true;
            const textElement = document.getElementById("text");

            setInterval(() => {
                if (growing) {
                    size++;
                    if (size >= 50) {
                        growing = false;
                        textElement.style.color = "blue";
                        textElement.textContent = "TEXT-SHRINKING";
                    }
                } else {
                    size--;
                    if (size <= 5) {
                        growing = true;
                        textElement.style.color = "red";
                        textElement.textContent = "TEXT-GROWING";
                    }
                }
                textElement.style.fontSize = size + "pt";
            }, 100);
        </script>
    </body>
    </html>

   <!-- Explanation:-->
	<!--1.	HTML:-->
	<!--•	A single div element with id="text" starts with a font size of 5pt and text “TEXT-GROWING”.-->
	<!--2.	JavaScript:-->
	<!--•	A setInterval function runs every 100ms.-->
	<!--•	The fontSize increases from 5pt to 50pt (TEXT-GROWING in red) and then decreases to 5pt (TEXT-SHRINKING in blue).-->
	<!--•	The growing boolean flag toggles the direction of size change.-->
	<!--3.	Inline Styling:-->
	<!--•	Font size and color are updated dynamically using style properties.-->

   <!-- Behavior:-->
	<!--•	The text grows from 5pt to 50pt in red.-->
	<!--•	On reaching 50pt, it changes to blue and shrinks back to 5pt.-->
	<!--•	This process repeats infinitely.-->`,

  6: `<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Squares and Cubes</title>
    </head>
    <body>
        <h2>Squares and Cubes Table</h2>
        <div id="table-container"></div>

        <script>
            let table = "<table border='1' cellspacing='0' cellpadding='10'><tr><th>Number</th><th>Square</th><th>Cube</th></tr>";
            for (let i = 0; i <= 10; i++) {
                table += '<tr><td>$'{i}</td><td>$'{i ** 2}</td><td>$'{i ** 3}</td></tr>';
            }
            table += "</table>";
            document.getElementById("table-container").innerHTML = table;
        </script>
    </body>
    </html>
   <!-- use tilda instead of single quotes '<tr><td>$'{i}</td><td>$'{i ** 2}</td><td>$'{i ** 3}</td></tr>'-->
   <!--	1.	HTML:-->
	<!--•	A div with id="table-container" where the dynamically created table will be displayed.-->
	<!--2.	JavaScript:-->
	<!--•	A string table is initialized with the table’s header row.-->
	<!--•	A for loop calculates the square and cube of numbers from 0 to 10.-->
	<!--•	Each result is added as a table row using template literals.-->
	<!--3.	Output:-->
	<!--•	A clean HTML table with borders that displays:-->
	<!--•	Number, Square, and Cube values.-->`,

  7: `<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>String and Number Functions</title>
    </head>
    <body>
        <h2>JavaScript Functions</h2>
        <p><strong>1. Left-most Vowel Position:</strong> <span id="vowelPosition"></span></p>
        <p><strong>2. Reversed Number:</strong> <span id="reversedNumber"></span></p>

        <script>
            // Function to find the left-most vowel position in a string
            function findVowelPosition(str) {
                const vowels = 'aeiouAEIOU';
                for (let i = 0; i < str.length; i++) {
                    if (vowels.indexOf(str[i]) !== -1) {
                        return i;  // Return position of first vowel
                    }
                }
                return -1; // No vowel found
            }

            // Function to reverse the digits of a number
            function reverseNumber(num) {
                return num.toString().split('').reverse().join('');
            }

            // Test the functions
            document.getElementById("vowelPosition").textContent = findVowelPosition("mvyan");
            document.getElementById("reversedNumber").textContent = reverseNumber(123123123);
        </script>
    </body>
    </html>
   <!-- Explanation:-->
	<!--1.	findVowelPosition(str):-->
	<!--•	Takes a string as a parameter.-->
	<!--•	Checks each character to see if it’s a vowel (using the string vowels).-->
	<!--•	Returns the position of the first vowel found, or -1 if no vowel is present.-->
	<!--2.	reverseNumber(num):-->
	<!--•	Takes a number as a parameter.-->
	<!--•	Converts it to a string, splits it into an array of characters, reverses the array, and then joins it back into a string.-->
	<!--3.	Example Tests:-->
	<!--•	For the string "hello", the first vowel (e) is found at position 1.-->
	<!--•	The number 12345 is reversed to 54321.-->`,

  8: `<!--Here is a simple webpage implementation of a search bar feature with a navigation bar. The search bar is positioned at the top-right corner of the navbar, and a search button is included next to it. The search will check for basic keywords using JavaScript.-->

  <!--HTML + CSS + JavaScript Code-->

  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Search Bar Feature</title>
      <style>
          /* Basic styling for the navbar */
          nav {
              background-color: #333;
              padding: 10px;
              display: flex;
              justify-content: space-between;
              align-items: center;
          }
          nav a {
              color: white;
              text-decoration: none;
              margin: 0 15px;
          }
          /* Search bar and button */
          .search-container {
              display: flex;
              align-items: center;
          }
          #searchBar {
              padding: 5px;
              font-size: 16px;
          }
          #searchButton {
              padding: 5px 10px;
              font-size: 16px;
              cursor: pointer;
          }
      </style>
  </head>
  <body>
      <!-- Navigation Bar -->
      <nav>
          <div>
              <a href="#">Home</a>
              <a href="#">About</a>
              <a href="#">Services</a>
              <a href="#">Contact</a>
          </div>
          <!-- Search Bar -->
          <div class="search-container">
              <input type="text" id="searchBar" placeholder="Search..." />
              <button id="searchButton">Search</button>
          </div>
      </nav>

      <!-- Search Results -->
      <div id="results"></div>

      <script>
          // Search functionality
          const searchButton = document.getElementById("searchButton");
          const searchBar = document.getElementById("searchBar");
          const resultsDiv = document.getElementById("results");

          const keywords = ["JavaScript", "HTML", "CSS", "Python", "Node.js", "React"];

          searchButton.addEventListener("click", function() {
              const query = searchBar.value.toLowerCase();
              const matches = keywords.filter(keyword => keyword.toLowerCase().includes(query));

              if (matches.length > 0) {
                  resultsDiv.innerHTML = "<h3>Results:</h3><ul>" + matches.map(match => '<li>$'{match}</li>').join("") + "</ul>";
              } else {
                  resultsDiv.innerHTML = "<h3>No results found</h3>";
              }
          });
      </script>
  </body>
  </html>

  <!--Explanation:-->
  <!--	1.	Navigation Bar:-->
  <!--	•	The navbar is styled with basic flexbox to align links to the left and the search bar to the right.-->
  <!--	•	The links (Home, About, Services, Contact) are created within an anchor tag (<a>).-->
  <!--	2.	Search Bar:-->
  <!--	•	The search bar is placed inside a div with the class search-container for alignment. It consists of an input field (<input>) and a button (<button>).-->
  <!--	•	The input field allows the user to type a search query.-->
  <!--	•	The button triggers the search functionality when clicked.-->
  <!--	3.	JavaScript Functionality:-->
  <!--	•	When the search button is clicked, the value from the search bar is compared against an array of basic keywords.-->
  <!--	•	The results are displayed in a <div> below the navbar. If the search query matches any keywords, they are shown in a list.-->
  <!--	•	If no match is found, a “No results found” message is displayed.-->

  <!--Features:-->
  <!--	•	Minimal Code: Simple structure with inline styles.-->
  <!--	•	Basic Search: Filters and displays the results based on predefined keywords like JavaScript, HTML, CSS, etc.-->
  <!--	•	Responsive: The layout adapts to the screen size because of flexbox.-->
  <!--use tilde instead of single quotes '<li>$'{match}</li>-->'`,

  9: `<!--Here is the minimal JavaScript code to design a simple calculator that performs the following operations: sum, product, difference, and quotient.-->

  <!--HTML + JavaScript Code-->

  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Simple Calculator</title>
  </head>
  <body>
      <h2>Simple Calculator</h2>
      <input type="number" id="num1" placeholder="Enter first number" />
      <input type="number" id="num2" placeholder="Enter second number" />
      <br><br>
      <button onclick="calculate('sum')">Sum</button>
      <button onclick="calculate('product')">Product</button>
      <button onclick="calculate('difference')">Difference</button>
      <button onclick="calculate('quotient')">Quotient</button>
      <h3 id="result"></h3>

      <script>
          function calculate(operation) {
              let num1 = parseFloat(document.getElementById('num1').value);
              let num2 = parseFloat(document.getElementById('num2').value);
              let result;

              if (operation === 'sum') {
                  result = num1 + num2;
              } else if (operation === 'product') {
                  result = num1 * num2;
              } else if (operation === 'difference') {
                  result = num1 - num2;
              } else if (operation === 'quotient') {
                  result = num2 !== 0 ? num1 / num2 : 'Error: Division by 0';
              }

              document.getElementById('result').innerText = 'Result: ' + result;
          }
      </script>
  </body>
  </html>

  <!--Explanation:-->
  <!--	1.	HTML:-->
  <!--	•	Two input fields (<input type="number">) for entering the numbers.-->
  <!--	•	Four buttons for performing the operations (Sum, Product, Difference, and Quotient).-->
  <!--	•	A <h3> element to display the result.-->
  <!--	2.	JavaScript:-->
  <!--	•	The calculate() function accepts an operation type (sum, product, difference, quotient) as a parameter.-->
  <!--	•	It retrieves the numbers from the input fields, performs the corresponding calculation, and displays the result in the <h3> tag.-->
  <!--	•	If division by zero is attempted, an error message is displayed.-->

  <!--Operations:-->
  <!--	•	Sum: Adds the two numbers.-->
  <!--	•	Product: Multiplies the two numbers.-->
  <!--	•	Difference: Subtracts the second number from the first.-->
  <!--	•	Quotient: Divides the first number by the second, handling division by zero.-->

  <!--This code provides a functional calculator with minimal code.-->`,

  10: `<!--Here is a minimal user registration website with Login, Signup, Login Success, Invalid Password Prompt, and Forgot Password pages using HTML, CSS, and JavaScript.-->

  <!--Complete Website Code-->

  <!--1. index.html (Login Page)-->

  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Login Page</title>
      <link rel="stylesheet" href="styles.css">
  </head>
  <body>
      <div class="container">
          <h2>Login</h2>
          <form id="loginForm">
              <input type="email" id="email" placeholder="Enter Email" required>
              <input type="password" id="password" placeholder="Enter Password" required>
              <button type="submit">Login</button>
          </form>
          <p><a href="signup.html">Sign Up</a> | <a href="forgot-password.html">Forgot Password?</a></p>
          <p id="error" style="color: red;"></p>
      </div>
      <script src="login.js"></script>
  </body>
  </html>

  <!--2. signup.html (Signup Page)-->

  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Signup Page</title>
      <link rel="stylesheet" href="styles.css">
  </head>
  <body>
      <div class="container">
          <h2>Sign Up</h2>
          <form id="signupForm">
              <input type="email" id="signupEmail" placeholder="Enter Email" required>
              <input type="password" id="signupPassword" placeholder="Enter Password" required>
              <input type="password" id="confirmPassword" placeholder="Confirm Password" required>
              <button type="submit">Sign Up</button>
          </form>
          <p>Already have an account? <a href="index.html">Login</a></p>
          <p id="signupError" style="color: red;"></p>
      </div>
      <script src="signup.js"></script>
  </body>
  </html>

  <!--3. forgot-password.html (Forgot Password Page)-->

  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Forgot Password</title>
      <link rel="stylesheet" href="styles.css">
  </head>
  <body>
      <div class="container">
          <h2>Forgot Password</h2>
          <form id="forgotPasswordForm">
              <input type="email" id="forgotEmail" placeholder="Enter Email" required>
              <button type="submit">Send Reset Link</button>
          </form>
          <p><a href="index.html">Back to Login</a></p>
          <p id="forgotError" style="color: red;"></p>
      </div>
      <script src="forgot-password.js"></script>
  </body>
  </html>

  <!--4. success.html (Login Success Page)-->

  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Login Success</title>
      <link rel="stylesheet" href="styles.css">
  </head>
  <body>
      <div class="container">
          <h2>Login Successful</h2>
          <p>Welcome to the website!</p>
          <a href="index.html">Logout</a>
      </div>
  </body>
  </html>

  5. styles.css (Common Styles)

  body {
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 0;
      background-color: #f4f4f4;
  }

  .container {
      max-width: 400px;
      margin: 50px auto;
      padding: 20px;
      background-color: white;
      border-radius: 8px;
      box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  }

  h2 {
      text-align: center;
  }

  input {
      width: 100%;
      padding: 10px;
      margin: 10px 0;
      border: 1px solid #ccc;
      border-radius: 4px;
  }

  button {
      width: 100%;
      padding: 10px;
      background-color: #4CAF50;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
  }

  button:hover {
      background-color: #45a049;
  }

  a {
      color: #007bff;
      text-decoration: none;
  }

  a:hover {
      text-decoration: underline;
  }

  6. login.js (Login Functionality)

  document.getElementById('loginForm').addEventListener('submit', function(e) {
      e.preventDefault();
      var email = document.getElementById('email').value;
      var password = document.getElementById('password').value;

      // Dummy validation
      if (email === "user@example.com" && password === "password123") {
          window.location.href = "success.html"; // Redirect to success page
      } else {
          document.getElementById('error').innerText = "Invalid email or password.";
      }
  });

  7. signup.js (Signup Functionality)

  document.getElementById('signupForm').addEventListener('submit', function(e) {
      e.preventDefault();
      var email = document.getElementById('signupEmail').value;
      var password = document.getElementById('signupPassword').value;
      var confirmPassword = document.getElementById('confirmPassword').value;

      if (password !== confirmPassword) {
          document.getElementById('signupError').innerText = "Passwords do not match!";
          return;
      }

      // Dummy registration
      alert("Sign Up Successful!");
      window.location.href = "index.html"; // Redirect to login page
  });

  8. forgot-password.js (Forgot Password Functionality)

  document.getElementById('forgotPasswordForm').addEventListener('submit', function(e) {
      e.preventDefault();
      var email = document.getElementById('forgotEmail').value;

      // Dummy email check
      if (email === "user@example.com") {
          alert("Password reset link sent!");
      } else {
          document.getElementById('forgotError').innerText = "Email not found.";
      }
  });

  Explanation:
	•	Login Page: Allows the user to log in with a predefined email and password. On successful login, the user is redirected to the success.html page.
	•	Signup Page: Lets the user create a new account by entering an email and password (with confirmation). After successful signup, the user is redirected to the Login page.
	•	Forgot Password Page: Provides a way for users to request a password reset link. A dummy check ensures the email exists.
	•	Login Success Page: Displays a success message upon successful login.
	•	JavaScript: Handles form submission, validation, and redirects.

  Notes:
	1.	The application uses basic form validation and dummy data (email: user@example.com, password: password123).
	2.	You can enhance the validation and integrate it with backend services for actual functionality.
	3.	This code is kept minimal for demonstration and learning purposes.`,
};

module.exports = psResponses;
