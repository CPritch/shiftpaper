# Print an optspec for argparse to handle cmd's options that are independent of any subcommand.
function __fish_shiftpaper_global_optspecs
    string join \n h/help V/version
end

function __fish_shiftpaper_needs_command
    # Figure out if the current invocation already has a command.
    set -l cmd (commandline -opc)
    set -e cmd[1]
    argparse -s (__fish_shiftpaper_global_optspecs) -- $cmd 2>/dev/null
    or return
    if set -q argv[1]
        # Also print the command, so this can be used to figure out what it is.
        echo $argv[1]
        return 1
    end
    return 0
end

function __fish_shiftpaper_using_subcommand
    set -l cmd (__fish_shiftpaper_needs_command)
    test -z "$cmd"
    and return 1
    contains -- $cmd[1] $argv
end

complete -c shiftpaper -n "__fish_shiftpaper_needs_command" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c shiftpaper -n "__fish_shiftpaper_needs_command" -s V -l version -d 'Print version'
complete -c shiftpaper -n "__fish_shiftpaper_needs_command" -f -a "fetch-model" -d 'Download a depth model to bake with'
complete -c shiftpaper -n "__fish_shiftpaper_needs_command" -f -a "set" -d 'Bake an image and make it the wallpaper'
complete -c shiftpaper -n "__fish_shiftpaper_needs_command" -f -a "slideshow" -d 'Bake several images and show them in turn'
complete -c shiftpaper -n "__fish_shiftpaper_needs_command" -f -a "transition" -d 'Show or change how one wallpaper changes into the next'
complete -c shiftpaper -n "__fish_shiftpaper_needs_command" -f -a "mode" -d 'Show or change the cursor tracking mode'
complete -c shiftpaper -n "__fish_shiftpaper_needs_command" -f -a "bake" -d 'Bake an image without setting it'
complete -c shiftpaper -n "__fish_shiftpaper_needs_command" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand fetch-model" -l list -d 'List the models that can be downloaded'
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand fetch-model" -s f -l force -d 'Re-download even if the files already exist'
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand fetch-model" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand set" -s m -l model -d 'The ONNX depth model to bake with. Defaults to $SHIFTPAPER_MODEL, then the one in config.toml. Whichever is used is saved for next time' -r -F
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand set" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand slideshow" -s i -l interval -d 'How long each image shows for, like 90s, 10m or 1h' -r
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand slideshow" -s m -l model -d 'The ONNX depth model to bake with. Defaults to $SHIFTPAPER_MODEL, then the one in config.toml' -r -F
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand slideshow" -s s -l shuffle -d 'Show the images in a random order, different each time round'
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand slideshow" -l stop -d 'Stop the slideshow, keeping the image it\'s showing'
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand slideshow" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand transition" -s s -l secs -d 'How long it takes, in seconds, like 3 or 1.5' -r
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand transition" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand mode" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand bake" -s o -l out -d 'Where to write the files. Defaults to the cache, ~/.cache/shiftpaper/wallpapers' -r -F
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand bake" -s m -l model -d 'The ONNX depth model to bake with. Defaults to $SHIFTPAPER_MODEL, then the one in config.toml' -r -F
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand bake" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand help; and not __fish_seen_subcommand_from fetch-model set slideshow transition mode bake help" -f -a "fetch-model" -d 'Download a depth model to bake with'
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand help; and not __fish_seen_subcommand_from fetch-model set slideshow transition mode bake help" -f -a "set" -d 'Bake an image and make it the wallpaper'
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand help; and not __fish_seen_subcommand_from fetch-model set slideshow transition mode bake help" -f -a "slideshow" -d 'Bake several images and show them in turn'
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand help; and not __fish_seen_subcommand_from fetch-model set slideshow transition mode bake help" -f -a "transition" -d 'Show or change how one wallpaper changes into the next'
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand help; and not __fish_seen_subcommand_from fetch-model set slideshow transition mode bake help" -f -a "mode" -d 'Show or change the cursor tracking mode'
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand help; and not __fish_seen_subcommand_from fetch-model set slideshow transition mode bake help" -f -a "bake" -d 'Bake an image without setting it'
complete -c shiftpaper -n "__fish_shiftpaper_using_subcommand help; and not __fish_seen_subcommand_from fetch-model set slideshow transition mode bake help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
