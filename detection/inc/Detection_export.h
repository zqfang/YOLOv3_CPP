
#ifndef DETECTION_EXPORT_H
#define DETECTION_EXPORT_H

#ifdef DETECTION_STATIC_DEFINE
#  define DETECTION_EXPORT
#  define DETECTION_NO_EXPORT
#else
#  ifndef DETECTION_EXPORT
#    ifdef detection_EXPORTS
        /* We are building this library */
#      define DETECTION_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define DETECTION_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef DETECTION_NO_EXPORT
#    define DETECTION_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef DETECTION_DEPRECATED
#  define DETECTION_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef DETECTION_DEPRECATED_EXPORT
#  define DETECTION_DEPRECATED_EXPORT DETECTION_EXPORT DETECTION_DEPRECATED
#endif

#ifndef DETECTION_DEPRECATED_NO_EXPORT
#  define DETECTION_DEPRECATED_NO_EXPORT DETECTION_NO_EXPORT DETECTION_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef DETECTION_NO_DEPRECATED
#    define DETECTION_NO_DEPRECATED
#  endif
#endif

#endif /* DETECTION_EXPORT_H */
