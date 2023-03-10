# This is a hack to prevent version errors in some environments
using PyCall
pyimport("h5py")

using Dates, Formatting, Logging

function metafmt(level::LogLevel, _module, group, id, file, line)
    @nospecialize
    color = Logging.default_logcolor(level)
    dt = format("[{:<23}] ", string(Dates.now()))
    
    prefix = string(dt, level == Logging.Warn ? "Warning" : string(level), ':')
    suffix::String = ""
    Logging.Info <= level < Logging.Warn && return color, prefix, suffix
    _module !== nothing && (suffix *= "$(_module)")
    if file !== nothing
        _module !== nothing && (suffix *= " ")
        suffix *= Base.contractuser(file)::String
        if line !== nothing
            suffix *= ":$(isa(line, UnitRange) ? "$(first(line))-$(last(line))" : line)"
        end
    end
    !isempty(suffix) && (suffix = "@ " * suffix)
    return color, prefix, suffix
end

global_logger(ConsoleLogger(meta_formatter=metafmt))

using MCStreamer
MCStreamer.main()
