# VHS Tape for Yevrah Terminal - Natural Language Query Demo
# https://github.com/charmbracelet/vhs

Output demos/demo1-natural-language.gif

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 800
Set Padding 20
Set Theme "Dracula"

Type "python main.py"
Enter
Sleep 2s

# Natural language query
Type "customer slip fall store florida last 3 years"
Enter
Sleep 5s

# Wait for results to display
Sleep 3s

# Select a case to analyze
Type "1"
Enter
Sleep 5s

# Wait to see analysis
Sleep 3s

# Exit
Type "exit"
Enter
Sleep 1s
