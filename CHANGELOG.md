## [1.0.0] - 2025-03-20
### Changes
- Initial release

## [1.0.1] - 2025-05-03
### Changes
#### Features
- ^ed testing script
- You can now set the voice similarity threshold for speaker detection in the transcriptor conditioning object

#### Bug fixes
- nova.py can now be imported from outside the Nova2 folder

## [1.1.0] - 2025-05-24
### Changes
#### Features
- `get_current_model()` in all inference engines has been replaced with the property `model`. You can still call `get_current_model`, but you will get a deprecation warning, as it will be removed in version 2.0.0.
- Tool calls made by the LLM will now be verified, whether they follow the correct structure laid out out in the metadata file. If not, the LLM will be informed about its incorrect call structure and the tool call will not be executed.

#### Bug fixes
- The elevenlabs inference engine now correctly reads the `similarity_boost` and `use_speaker_boost` parameters from the conditioning object.

#### General improvements
- The following classes now use a Singleton pattern: `ContextManager`, `MemoryEmbeddingDatabaseManager`, `VoiceDatabaseManager`, `ToolManager`
- Context data is now no longer being written to the disk for every change. It is instead updated in regular intervals (every 2 minutes), and when the program exits.
- Type hinting now follows the pydantic standards.
- The codebase now only uses absolute import paths, instead of relative ones.
- More tests have been added.