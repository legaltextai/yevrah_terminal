# VHS Tape for Yevrah Terminal - Jurisdiction Specific Search Demo
# https://github.com/charmbracelet/vhs

Output demos/demo3-jurisdiction-search.gif

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 800
Set Padding 20
Set Theme "Dracula"

Type "python main.py"
Enter
Sleep 2s

# Jurisdiction-specific query
Type "constitutional rights, supreme court, last 2 years"
Enter
Sleep 5s

# Wait for results to display
Sleep 3s

# Show results
Sleep 2s

# Exit
Type "exit"
Enter
Sleep 1s
