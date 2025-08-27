HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Real or Fake Checker</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css">
    <style>
        body { background-color: #7bbed5; height: 100vh; display: flex; justify-content: center; align-items: center; }
        .container { background: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); max-width: 500px; width: 100%; }
        h1 { font-size: 2.5rem; color: #290f28; text-align: center; }
        .alert { text-align: center; font-size: 1.2rem; }
        .result-label { font-weight: bold; font-size: 1.4rem; display: block; margin-bottom: 10px; }
        .form-control { height: 45px; }
        .btn-success { color: #0e133a; width: 100%; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Fake News Checker</h1>
        <form method="POST" action="/check">
            <div class="mb-3">
                <label for="news" class="form-label">Enter News Headline</label>
                <input type="text" id="news" name="news" class="form-control" required>
            </div>
            <button type="submit" class="btn btn-success">Check</button>
        </form>
        {% if result %}
        <div class="alert alert-info mt-4" role="alert">
            <span class="result-label">{{ result.split('(')[0] }}</span>
            <span>{{ result.split('(')[1] }}</span>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""
