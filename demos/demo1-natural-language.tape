# VHS Tape for Yevrah Terminal - Natural Language Query Demo
# https://github.com/charmbracelet/vhs

Output demo1-natural-language.gif

Set Shell "bash"
Set FontSize 16
Set Width 1400
Set Height 900
Set Padding 20
Set Theme "Catppuccin Mocha"
Set CursorBlink false

# Start the application
Type@500ms "python main.py"
Enter

# Wait 7 seconds for app to be ready for input
Sleep 7s

# Pause to show the welcome message
Sleep 3s

# Enter the search query
Type@500ms "customer fell in a store, slip-and-fall, florida, last 3 years"
Enter

# Wait 7 seconds for search results to appear
Sleep 7s

# Scroll up to show search parameters and keyword/semantic tags
Up@1s 10
Sleep 2s

# Scroll back down to see all results
Down@500ms 10
Sleep 2s

# Select case #3 to analyze
Type "3"
Enter

# Wait 10 seconds for analysis to appear
Sleep 10s

# Let user read some of the analysis
Sleep 3s

# Exit
Type "exit"
Enter
Sleep 1s
